import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hydraulic_faults.contract import CHANNELS, CONDITIONS, WINDOW_SIZE, InputError

SCHEMA_VERSION = 1
MAX_MODEL_BYTES = 65_536
FEATURE_COUNT = len(CHANNELS) * 5


def window_features(windows: np.ndarray) -> np.ndarray:
    """Compute the same causal features in training and streaming inference."""
    return np.concatenate((
        windows.mean(axis=-2),
        windows.std(axis=-2),
        windows.min(axis=-2),
        windows.max(axis=-2),
        windows[..., -1, :] - windows[..., 0, :],
    ), axis=-1)


@dataclass(frozen=True)
class LinearHead:
    labels: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray


@dataclass(frozen=True)
class Classifier:
    mean: np.ndarray
    scale: np.ndarray
    heads: tuple[LinearHead, ...]

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.mean) / self.scale
        return np.stack([
            head.labels[(standardized @ head.coefficients.T + head.intercept).argmax(axis=-1)]
            for head in self.heads
        ], axis=-1)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "channels": [channel.name for channel in CHANNELS],
            "window_size": WINDOW_SIZE,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "heads": {
                name: {
                    "labels": head.labels.tolist(),
                    "coefficients": head.coefficients.tolist(),
                    "intercept": head.intercept.tolist(),
                }
                for name, head in zip(CONDITIONS, self.heads, strict=True)
            },
        }


def _numeric_array(data, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(data)
    if (
        values.shape != shape or values.dtype.kind not in "iuf"
        or not np.isfinite(values).all() or ((values < -1e12) | (values > 1e12)).any()
    ):
        raise InputError("invalid model parameters")
    return values.astype(float)


def load_model(path: Path) -> Classifier:
    with path.open("rb") as model_file:
        raw = model_file.read(MAX_MODEL_BYTES + 1)
    if len(raw) > MAX_MODEL_BYTES:
        raise InputError("model exceeds size limit")
    try:
        data = json.loads(raw.decode("utf-8"))
        if (
            data["schema_version"] != SCHEMA_VERSION
            or data["channels"] != [channel.name for channel in CHANNELS]
            or data["window_size"] != WINDOW_SIZE
            or data["heads"].keys() != CONDITIONS.keys()
        ):
            raise InputError("incompatible model schema")
        mean = _numeric_array(data["mean"], (FEATURE_COUNT,))
        scale = _numeric_array(data["scale"], (FEATURE_COUNT,))
        if (scale < 1e-12).any():
            raise InputError("invalid model scale")
        heads = []
        for name, conditions in CONDITIONS.items():
            head = data["heads"][name]
            labels = head["labels"]
            if labels != sorted(conditions):
                raise InputError("incompatible condition labels")
            heads.append(LinearHead(
                np.asarray(labels, dtype=int),
                _numeric_array(head["coefficients"], (len(labels), FEATURE_COUNT)),
                _numeric_array(head["intercept"], (len(labels),)),
            ))
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError, RecursionError) as exc:
        raise InputError("invalid model artifact") from exc
    return Classifier(mean, scale, tuple(heads))
