import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hydraulic_faults.contract import CHANNELS, CONDITIONS, NORMAL_CODES, WINDOW_SIZE, InputError

LINEAR_SCHEMA_VERSION = 2
SCHEMA_VERSION = 3
MAX_MODEL_BYTES = 65_536
FEATURE_COUNT = len(CHANNELS) * 5
MAX_TREE_DEPTH = 8
MAX_TREE_NODES = 2 ** (MAX_TREE_DEPTH + 1) - 1


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

    def predict(self, standardized: np.ndarray) -> np.ndarray:
        logits = standardized @ self.coefficients.T + self.intercept
        return self.labels[logits.argmax(axis=-1)]

    def to_dict(self) -> dict:
        return {
            "labels": self.labels.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept.tolist(),
        }


@dataclass(frozen=True)
class TreeHead:
    labels: np.ndarray
    children: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    class_indices: np.ndarray

    def predict(self, standardized: np.ndarray) -> np.ndarray:
        # Match scikit-learn's float32 input and float64 split thresholds.
        values = standardized.reshape(-1, FEATURE_COUNT).astype(np.float32)
        nodes = np.zeros(len(values), dtype=int)
        active = np.arange(len(values))
        while active.size:
            parents = nodes[active]
            branches = self.children[parents, 0] >= 0
            active, parents = active[branches], parents[branches]
            go_right = values[active, self.features[parents]] > self.thresholds[parents]
            nodes[active] = self.children[parents, go_right.astype(int)]
        return self.labels[self.class_indices[nodes]].reshape(standardized.shape[:-1])

    def to_dict(self) -> dict:
        return {
            "labels": self.labels.tolist(),
            "tree": {
                "children": self.children.tolist(),
                "features": self.features.tolist(),
                "thresholds": self.thresholds.tolist(),
                "class_indices": self.class_indices.tolist(),
            },
        }


@dataclass(frozen=True)
class Classifier:
    mean: np.ndarray
    scale: np.ndarray
    fault_head: LinearHead
    heads: tuple[LinearHead | TreeHead, ...]

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        standardized = (features - self.mean) / self.scale
        faulty = self.fault_head.predict(standardized).astype(bool)
        conditions = np.stack([head.predict(standardized) for head in self.heads], axis=-1)
        return faulty, np.where(faulty[..., None], conditions, NORMAL_CODES)

    def to_dict(self) -> dict:
        version = SCHEMA_VERSION if any(isinstance(head, TreeHead) for head in self.heads) else LINEAR_SCHEMA_VERSION
        return {
            "schema_version": version,
            "channels": [channel.name for channel in CHANNELS],
            "window_size": WINDOW_SIZE,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fault_head": self.fault_head.to_dict(),
            "heads": {
                name: head.to_dict()
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


def _load_linear_head(data: dict, labels: list[int]) -> LinearHead:
    if data["labels"] != labels:
        raise InputError("incompatible model labels")
    return LinearHead(
        np.asarray(labels, dtype=int),
        _numeric_array(data["coefficients"], (len(labels), FEATURE_COUNT)),
        _numeric_array(data["intercept"], (len(labels),)),
    )


def _index_array(data, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(data)
    if values.shape != shape or values.dtype.kind not in "iu":
        raise InputError("invalid tree indices")
    return values


def _load_tree_head(data: dict, labels: list[int]) -> TreeHead:
    if data["labels"] != labels:
        raise InputError("incompatible model labels")
    tree = data["tree"]
    count = len(tree["children"])
    if not 1 <= count <= MAX_TREE_NODES:
        raise InputError("invalid tree size")
    children = _index_array(tree["children"], (count, 2))
    features = _index_array(tree["features"], (count,))
    thresholds = _numeric_array(tree["thresholds"], (count,))
    classes = _index_array(tree["class_indices"], (count,))
    branches = children[:, 0] != -1
    parents = np.flatnonzero(branches)
    child_nodes = children[branches]
    if (
        (children[~branches] != -1).any()
        or (child_nodes <= parents[:, None]).any() or (child_nodes >= count).any()
        or ((features[branches] < 0) | (features[branches] >= FEATURE_COUNT)).any()
        or ((classes < 0) | (classes >= len(labels))).any()
    ):
        raise InputError("invalid tree nodes")
    if (np.bincount(child_nodes.ravel().astype(int), minlength=count)[1:] != 1).any():
        raise InputError("each tree node except the root must have exactly one parent")
    depths = [0] * count
    for parent, (left, right) in zip(parents, child_nodes, strict=True):
        depth = depths[parent] + 1
        if depth > MAX_TREE_DEPTH:
            raise InputError("tree exceeds depth limit")
        depths[left] = depths[right] = depth
    return TreeHead(np.asarray(labels, dtype=int), children, features, thresholds, classes)


def load_model(path: Path) -> Classifier:
    with path.open("rb") as model_file:
        raw = model_file.read(MAX_MODEL_BYTES + 1)
    if len(raw) > MAX_MODEL_BYTES:
        raise InputError("model exceeds size limit")
    try:
        data = json.loads(raw.decode("utf-8"))
        version = data["schema_version"]
        if (
            version not in (LINEAR_SCHEMA_VERSION, SCHEMA_VERSION)
            or data["channels"] != [channel.name for channel in CHANNELS]
            or data["window_size"] != WINDOW_SIZE
            or data["heads"].keys() != CONDITIONS.keys()
        ):
            raise InputError("incompatible model schema")
        mean = _numeric_array(data["mean"], (FEATURE_COUNT,))
        scale = _numeric_array(data["scale"], (FEATURE_COUNT,))
        if (scale < 1e-12).any():
            raise InputError("invalid model scale")
        fault_head = _load_linear_head(data["fault_head"], [0, 1])
        heads = []
        for name, conditions in CONDITIONS.items():
            head = data["heads"][name]
            loader = _load_tree_head if version == SCHEMA_VERSION and "tree" in head else _load_linear_head
            heads.append(loader(head, sorted(conditions)))
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError, RecursionError) as exc:
        raise InputError("invalid model artifact") from exc
    return Classifier(mean, scale, fault_head, tuple(heads))
