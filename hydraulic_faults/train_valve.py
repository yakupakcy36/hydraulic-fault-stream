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
    FEATURE_COUNT,
    VALVE_LABELS,
    ValveModel,
    load_valve_captures,
    load_valve_model,
    transition_features,
)


def _fit(features: np.ndarray, targets: np.ndarray) -> ValveModel:
    if set(targets) != set(VALVE_LABELS):
        raise InputError("valve training needs a complete transition for every valve condition")
    scale = StandardScaler().fit(features)
    lda = LinearDiscriminantAnalysis().fit(scale.transform(features), targets)
    return ValveModel(scale.mean_, scale.scale_, LinearHead(lda.classes_, lda.coef_, lda.intercept_))


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


def train_valve(archive: Path, model_path: Path) -> dict:
    if archive.resolve() == model_path.resolve():
        raise InputError("model path must differ from the input archive")
    with archive.open("rb") as data:
        if os.fstat(data.fileno()).st_size > MAX_ARCHIVE_BYTES:
            raise InputError("input archive exceeds size limit")
        archive_sha = hashlib.file_digest(data, "sha256").hexdigest()
    captures, profiles = load_valve_captures(archive)
    features = np.zeros((len(profiles), FEATURE_COUNT))
    available = np.zeros(len(profiles), dtype=bool)
    for index, capture in enumerate(captures):
        result = transition_features(capture)
        if result is not None:
            features[index], _ = result
            available[index] = True
    targets = profiles[:, VALVE_INDEX]

    def fit(indices):
        selected = indices[available[indices]]
        return _fit(features[selected], targets[selected])

    def evaluate(model, indices):
        predicted = np.full(len(indices), -1)
        valid = available[indices]
        predicted[valid] = model.predict(features[indices[valid]])
        return _measure(targets[indices], predicted)

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
        "feature_spec": model.to_dict()["feature_spec"],
        "training_captures": int(available[train].sum()),
        "stable_holdout": evaluate(model, test),
        "development_folds": folds,
        "archive_guard_check": guard_report,
        "unstable_nominal_labels": evaluate(model, np.flatnonzero(~stable)) if (~stable).any() else None,
    }
    _save(model, model_path, features[test[available[test]]])
    return report
