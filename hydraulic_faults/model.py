import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hydraulic_faults.contract import CHANNELS, CONDITIONS, NORMAL_CODES, WINDOW_SIZE, InputError

LINEAR_SCHEMA_VERSION = 2
TREE_SCHEMA_VERSION = 3
SCHEMA_VERSION = 4
MAX_MODEL_BYTES = 65_536
FEATURE_COUNT = len(CHANNELS) * 5
MAX_TREE_DEPTH = 8
MAX_TREE_NODES = 2 ** (MAX_TREE_DEPTH + 1) - 1
VALVE_INDEX = tuple(CONDITIONS).index("valve")
VALVE_NEAR_FAILURE = next(code for code, label in CONDITIONS["valve"].items() if label == "near_failure")


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

    def logits(self, standardized: np.ndarray) -> np.ndarray:
        return standardized @ self.coefficients.T + self.intercept

    def predict(self, standardized: np.ndarray) -> np.ndarray:
        return self.labels[self.logits(standardized).argmax(axis=-1)]

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
    valve_alpha: float | None = None

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        faulty, conditions, _ = self.predict_with_raw_conditions(features)
        return faulty, conditions

    def predict_with_raw_conditions(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the fault decision, gated conditions and raw component predictions."""
        faulty, conditions, raw, _ = self.predict_step(features)
        return faulty, conditions, raw

    def predict_step(
        self, features: np.ndarray, previous_valve_scores: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Evaluate one window per machine or cycle and return the updated valve evidence."""
        standardized = (features - self.mean) / self.scale
        faulty = self.fault_head.predict(standardized).astype(bool)
        evidence = None
        component_codes = []
        for index, head in enumerate(self.heads):
            if index == VALVE_INDEX and self.valve_alpha is not None:
                current = head.logits(standardized)
                evidence = current
                instant = head.labels[current.argmax(axis=-1)]
                valve_code = instant
                if previous_valve_scores is not None:
                    evidence = self.valve_alpha * current + (1 - self.valve_alpha) * previous_valve_scores
                    averaged = head.labels[evidence.argmax(axis=-1)]
                    # History can reinforce a critical diagnosis but cannot hide an immediate one.
                    valve_code = np.where(instant == VALVE_NEAR_FAILURE, instant, averaged)
                component_codes.append(valve_code)
            else:
                component_codes.append(head.predict(standardized))
        raw = np.stack(component_codes, axis=-1)
        return faulty, np.where(faulty[..., None], raw, NORMAL_CODES), raw, evidence

    def predict_cycles(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate arrays shaped (cycles, windows, features) with fresh evidence per cycle."""
        if self.valve_alpha is None or not features.shape[1]:
            return self.predict_with_raw_conditions(features)
        faults, conditions, raw_conditions = [], [], []
        evidence = None
        for window in np.moveaxis(features, 1, 0):
            faulty, gated, raw, evidence = self.predict_step(window, evidence)
            faults.append(faulty)
            conditions.append(gated)
            raw_conditions.append(raw)
        return np.stack(faults, axis=1), np.stack(conditions, axis=1), np.stack(raw_conditions, axis=1)

    def to_dict(self) -> dict:
        version = TREE_SCHEMA_VERSION if any(isinstance(head, TreeHead) for head in self.heads) else LINEAR_SCHEMA_VERSION
        data = {
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
        if self.valve_alpha is not None:
            data["schema_version"] = SCHEMA_VERSION
            data["valve_alpha"] = self.valve_alpha
        return data


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
            version not in (LINEAR_SCHEMA_VERSION, TREE_SCHEMA_VERSION, SCHEMA_VERSION)
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
            loader = _load_tree_head if version >= TREE_SCHEMA_VERSION and "tree" in head else _load_linear_head
            heads.append(loader(head, sorted(conditions)))
        valve_alpha = None
        if version == SCHEMA_VERSION:
            valve_alpha = data["valve_alpha"]
            if type(valve_alpha) not in (int, float) or not 0 < valve_alpha < 1:
                raise InputError("invalid valve evidence weight")
            if not isinstance(heads[VALVE_INDEX], LinearHead):
                raise InputError("valve evidence requires a linear head")
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError, RecursionError) as exc:
        raise InputError("invalid model artifact") from exc
    return Classifier(mean, scale, fault_head, tuple(heads), valve_alpha)
