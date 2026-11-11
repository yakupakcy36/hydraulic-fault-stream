import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from hydraulic_faults.contract import NORMAL_CODES, InputError
from hydraulic_faults.dataset import MAX_ARCHIVE_BYTES
from hydraulic_faults.model import VALVE_INDEX, VALVE_NEAR_FAILURE, LinearHead
from hydraulic_faults.train import ARCHIVE_GUARD_CYCLES, _guarded_train_cycles, split_cycles
from hydraulic_faults.valve import (
    CAPTURE_SAMPLES,
    FEATURE_COUNT,
    SAMPLE_RATE_HZ,
    VALVE_LABELS,
    ValveModel,
    load_valve_captures,
    load_valve_model,
    transition_features,
)

PS1_TEST_OFFSETS = (-2, -1, 1, 2)
SKEW_CONTEXT_SAMPLES = max(map(abs, PS1_TEST_OFFSETS))


def _extract_features(captures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((len(captures), FEATURE_COUNT))
    available = np.zeros(len(captures), dtype=bool)
    for index, capture in enumerate(captures):
        result = transition_features(capture)
        if result is not None:
            features[index], _ = result
            available[index] = True
    return features, available


def _fit(features: np.ndarray, targets: np.ndarray) -> ValveModel:
    if set(targets) != set(VALVE_LABELS):
        raise InputError("valve training needs a complete transition for every valve condition")
    scale = StandardScaler().fit(features)
    lda = LinearDiscriminantAnalysis().fit(scale.transform(features), targets)
    return ValveModel(scale.mean_, scale.scale_, LinearHead(lda.classes_, lda.coef_, lda.intercept_))


def augmentation_features(captures: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed synthetic training perturbations; select training indices before fitting."""
    rng = np.random.default_rng(732)
    variants = [captures * 0.98, captures * 1.02]
    variants.extend(captures + rng.normal(0, 0.5, captures.shape) for _ in range(2))
    return [_extract_features(values) for values in variants]


def _measure(targets: np.ndarray, predictions: np.ndarray) -> dict:
    critical = targets == VALVE_NEAR_FAILURE
    return {
        "captures": len(targets),
        "unavailable": int((predictions == -1).sum()),
        "correct": int((targets == predictions).sum()),
        "macro_f1": round(f1_score(targets, predictions, labels=np.unique(targets), average="macro", zero_division=0), 4),
        "critical_captures": int(critical.sum()),
        "critical_correct": int((critical & (predictions == VALVE_NEAR_FAILURE)).sum()),
        "false_critical": int((~critical & (predictions == VALVE_NEAR_FAILURE)).sum()),
    }


def _evaluate(model: ValveModel, targets: np.ndarray, features: np.ndarray, available: np.ndarray) -> dict:
    predicted = np.full(len(targets), -1)
    predicted[available] = model.predict(features[available])
    return _measure(targets, predicted)


def _relative_skew_check(model: ValveModel, targets: np.ndarray, captures: np.ndarray) -> list[dict]:
    start = SKEW_CONTEXT_SAMPLES
    end = start + CAPTURE_SAMPLES
    shifted = captures[:, :, start:end].copy()
    results = []
    for offset in PS1_TEST_OFFSETS:
        shifted[:, 0] = captures[:, 0, start + offset:end + offset]
        features, available = _extract_features(shifted)
        results.append({
            "ps1_source_offset_samples": offset,
            "ps1_source_offset_ms": offset * (1_000 // SAMPLE_RATE_HZ),
            **_evaluate(model, targets, features, available),
        })
    return results


def _save(model: ValveModel, path: Path, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(model.to_dict(), stream, allow_nan=False)
        if not np.array_equal(load_valve_model(temporary).predict(features), model.predict(features)):
            raise InputError("exported valve model does not reproduce holdout predictions")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train_valve(archive: Path, model_path: Path, *, method: str = "augmented_lda") -> dict:
    if method not in ("lda", "augmented_lda"):
        raise InputError("unknown valve training method")
    if archive.resolve() == model_path.resolve():
        raise InputError("model path must differ from the input archive")
    with archive.open("rb") as data:
        if os.fstat(data.fileno()).st_size > MAX_ARCHIVE_BYTES:
            raise InputError("input archive exceeds size limit")
        archive_sha = hashlib.file_digest(data, "sha256").hexdigest()
    captures, profiles = load_valve_captures(archive, context_samples=SKEW_CONTEXT_SAMPLES)
    start = SKEW_CONTEXT_SAMPLES
    features, available = _extract_features(captures[:, :, start:start + CAPTURE_SAMPLES])
    targets = profiles[:, VALVE_INDEX]
    variants = [(features, available)]
    if method == "augmented_lda":
        stable_indices = np.flatnonzero(profiles[:, 4] == 0)
        for values, usable in augmentation_features(captures[stable_indices, :, start:start + CAPTURE_SAMPLES]):
            full_values, full_usable = np.zeros_like(features), np.zeros_like(available)
            full_values[stable_indices], full_usable[stable_indices] = values, usable
            variants.append((full_values, full_usable))

    def fit(indices):
        xs, ys = [], []
        for values, usable in variants:
            selected = indices[usable[indices]]
            xs.append(values[selected])
            ys.append(targets[selected])
        return _fit(np.concatenate(xs), np.concatenate(ys))

    def evaluate(model, indices):
        return _evaluate(model, targets[indices], features[indices], available[indices])

    train, test = split_cycles(profiles)
    model = fit(train)
    stable = profiles[:, 4] == 0
    healthy = (profiles[:, :4] == NORMAL_CODES).all(axis=1)
    stable_healthy = np.flatnonzero(stable & healthy)
    stable_fault = np.flatnonzero(stable & ~healthy)
    _, groups = np.unique(profiles[stable_fault, :4], axis=0, return_inverse=True)
    folds = []
    if len(np.unique(groups)) >= 4:
        splitter = GroupKFold(n_splits=4, shuffle=True, random_state=17)
        healthy_test = test[healthy[test]]
        healthy_train = stable_healthy[~np.isin(stable_healthy, healthy_test)]
        for fit_idx, eval_idx in splitter.split(stable_fault, groups=groups):
            fold_train = np.r_[stable_fault[fit_idx], healthy_train]
            fold_test = np.r_[stable_fault[eval_idx], healthy_test]
            if set(targets[fold_train[available[fold_train]]]) != set(VALVE_LABELS):
                folds.append({"skipped": "some valve conditions lack complete training captures"})
                continue
            folds.append(evaluate(fit(fold_train), fold_test))
    guarded = _guarded_train_cycles(profiles, train, test)
    guard_report = {"guard_cycles": ARCHIVE_GUARD_CYCLES, "train_cycles": len(guarded)}
    if set(targets[guarded[available[guarded]]]) == set(VALVE_LABELS):
        guard_report["holdout"] = evaluate(fit(guarded), test)
    else:
        guard_report["skipped"] = "some valve conditions lack complete guarded training captures"
    report = {
        "dataset_sha256": archive_sha,
        "training_method": method,
        "augmentation": {"seed": 732, "gain_factors": [0.98, 1.02], "noise_std_bar": 0.5, "noise_replicates": 2}
        if method == "augmented_lda" else None,
        "feature_spec": model.to_dict()["feature_spec"],
        "training_captures": int(available[train].sum()),
        "stable_holdout": evaluate(model, test),
        "stable_holdout_relative_skew": _relative_skew_check(model, targets[test], captures[test]),
        "development_folds": folds,
        "archive_guard_check": guard_report,
        "unstable_nominal_labels": evaluate(model, np.flatnonzero(~stable)) if (~stable).any() else None,
    }
    _save(model, model_path, features[test[available[test]]])
    return report
