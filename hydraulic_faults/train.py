import hashlib
import json
import os
import platform
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from hydraulic_faults.contract import CONDITIONS, NORMAL_CODES, WINDOW_SIZE, InputError
from hydraulic_faults.dataset import MAX_ARCHIVE_BYTES, load_cycles
from hydraulic_faults.model import MAX_TREE_DEPTH, Classifier, LinearHead, TreeHead, load_model, window_features

SEED = 42
TEST_FRACTION = 0.2
ACCUMULATOR_INDEX = tuple(CONDITIONS).index("accumulator")


def split_cycles(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stable = np.flatnonzero(profiles[:, 4] == 0)
    healthy = (profiles[stable, :4] == NORMAL_CODES).all(axis=1)
    healthy_cycles = stable[healthy]
    fault_cycles = stable[~healthy]
    fault_profiles, groups = np.unique(profiles[fault_cycles, :4], axis=0, return_inverse=True)
    if len(healthy_cycles) < 2 or len(fault_profiles) < 2:
        raise InputError("evaluation requires at least two stable healthy cycles and two distinct fault profiles")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
    train_idx, test_idx = next(splitter.split(fault_cycles, groups=groups))
    # The source records a single run of healthy cycles. Keep its tail for a smoke check.
    healthy_test_count = max(1, round(len(healthy_cycles) * TEST_FRACTION))
    train = np.sort(np.r_[fault_cycles[train_idx], healthy_cycles[:-healthy_test_count]])
    test = np.sort(np.r_[fault_cycles[test_idx], healthy_cycles[-healthy_test_count:]])
    for column, (name, labels) in enumerate(CONDITIONS.items()):
        if set(profiles[train, column]) != labels.keys() or set(profiles[test, column]) != labels.keys():
            raise InputError(f"grouped train and test must each cover all {name} conditions")
    return train, test


def _cycle_detection_metrics(profiles: np.ndarray, predicted_faults: np.ndarray) -> dict:
    windows_by_cycle = predicted_faults.reshape(len(profiles), -1)
    faulty = (profiles[:, :4] != NORMAL_CODES).any(axis=1)
    alerted = windows_by_cycle.any(axis=1)
    _, groups = np.unique(profiles[faulty, :4], axis=0, return_inverse=True)
    profile_recall = (
        np.bincount(groups, weights=windows_by_cycle[faulty].mean(axis=1)) / np.bincount(groups)
    )
    return {
        "fault_cycles": int(faulty.sum()),
        "fault_cycles_alerted": int(alerted[faulty].sum()),
        "healthy_cycles_alerted": int(alerted[~faulty].sum()),
        "fault_profiles": len(profile_recall),
        "fault_profiles_without_alert": int((profile_recall == 0).sum()),
        "macro_profile_window_recall": round(float(profile_recall.mean()), 4),
    }


def _window_examples(cycles: np.ndarray, profiles: np.ndarray):
    windows = np.lib.stride_tricks.sliding_window_view(cycles, WINDOW_SIZE, axis=1)
    features = window_features(windows.swapaxes(-1, -2))
    return features.reshape(-1, features.shape[-1]), np.repeat(profiles[:, :4], features.shape[1], axis=0)


def fit_classifier(features: np.ndarray, targets: np.ndarray) -> Classifier:
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    heads = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        faulty = (targets != NORMAL_CODES).any(axis=1).astype(int)
        detector = LogisticRegression(max_iter=2_000, class_weight="balanced").fit(standardized, faulty)
        for column in range(len(CONDITIONS)):
            model = LogisticRegression(max_iter=2_000, class_weight="balanced")
            model.fit(standardized, targets[:, column])
            heads.append(LinearHead(model.classes_, model.coef_, model.intercept_))
    # A binary logistic regression returns one logit. Use class 0 as the zero-logit reference.
    fault_head = LinearHead(
        detector.classes_,
        np.vstack((np.zeros_like(detector.coef_), detector.coef_)),
        np.r_[0.0, detector.intercept_],
    )
    return Classifier(scaler.mean_, scaler.scale_, fault_head, tuple(heads))


def _fit_accumulator_tree(standardized: np.ndarray, labels: np.ndarray) -> TreeHead:
    estimator = DecisionTreeClassifier(
        max_depth=MAX_TREE_DEPTH, min_samples_leaf=56, class_weight="balanced", random_state=SEED,
    ).fit(standardized, labels)
    tree = estimator.tree_
    return TreeHead(
        estimator.classes_,
        np.column_stack((tree.children_left, tree.children_right)),
        tree.feature,
        tree.threshold,
        tree.value[:, 0, :].argmax(axis=1),
    )


def _save_model(classifier: Classifier, path: Path, features: np.ndarray, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, encoding="utf-8", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(classifier.to_dict(), output, allow_nan=False)
        exported = load_model(temporary_path).predict(features)
        if not all(np.array_equal(a, b) for a, b in zip(exported, predictions, strict=True)):
            raise InputError("exported model does not reproduce predictions")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def train_and_evaluate(archive: Path, model_path: Path) -> dict:
    if archive.resolve() == model_path.resolve():
        raise InputError("model path must differ from the input archive")
    with archive.open("rb") as data:
        if os.fstat(data.fileno()).st_size > MAX_ARCHIVE_BYTES:
            raise InputError("input archive exceeds size limit")
        archive_sha = hashlib.file_digest(data, "sha256").hexdigest()
    cycles, profiles = load_cycles(archive)
    train, test = split_cycles(profiles)
    train_x, train_y = _window_examples(cycles[train], profiles[train])
    test_x, test_y = _window_examples(cycles[test], profiles[test])
    linear_model = fit_classifier(train_x, train_y)
    tree = _fit_accumulator_tree(
        (train_x - linear_model.mean) / linear_model.scale, train_y[:, ACCUMULATOR_INDEX],
    )
    heads = list(linear_model.heads)
    heads[ACCUMULATOR_INDEX] = tree
    model = replace(linear_model, heads=tuple(heads))
    predicted_faults, predicted = model.predict(test_x)
    _, linear_predictions = linear_model.predict(test_x)
    baseline = np.column_stack([
        np.full(len(test_y), np.bincount(train_y[:, column]).argmax())
        for column in range(len(CONDITIONS))
    ])
    faults = (test_y != NORMAL_CODES).any(axis=1)
    tn, fp, fn, tp = confusion_matrix(faults, predicted_faults, labels=[False, True]).ravel()
    component_metrics = {}
    for column, (name, labels) in enumerate(CONDITIONS.items()):
        expected = test_y[:, column]
        guess = predicted[:, column]
        codes = sorted(labels)
        component_metrics[name] = {
            "label_order": [labels[code] for code in codes],
            "macro_f1": round(f1_score(expected, guess, average="macro"), 4),
            "majority_baseline_macro_f1": round(
                f1_score(expected, baseline[:, column], labels=codes, average="macro", zero_division=0), 4,
            ),
            "confusion_matrix": confusion_matrix(expected, guess, labels=codes).tolist(),
        }
    component_metrics["accumulator"]["linear_baseline_macro_f1"] = round(
        f1_score(test_y[:, ACCUMULATOR_INDEX], linear_predictions[:, ACCUMULATOR_INDEX], average="macro"), 4,
    )
    _save_model(model, model_path, test_x, (predicted_faults, predicted))
    train_fault_cycles = (profiles[train, :4] != NORMAL_CODES).any(axis=1)
    return {
        "dataset_sha256": archive_sha,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "seed": SEED,
        "split_protocol": "grouped_fault_profiles_with_healthy_tail",
        "target_test_profile_fraction": TEST_FRACTION,
        "train_cycles": len(train),
        "train_fault_profiles": len(np.unique(profiles[train[train_fault_cycles], :4], axis=0)),
        "test_cycles": len(test),
        "normal_test_cycles": int((profiles[test, :4] == NORMAL_CODES).all(axis=1).sum()),
        "excluded_unstable_cycles": len(profiles) - len(train) - len(test),
        "test_windows": len(test_y),
        "fault_detection": {
            "precision": round(precision_score(faults, predicted_faults, zero_division=0), 4),
            "recall": round(recall_score(faults, predicted_faults, zero_division=0), 4),
            "always_fault_baseline_precision": round(float(faults.mean()), 4),
            "true_negatives": int(tn), "false_positives": int(fp),
            "false_negatives": int(fn), "true_positives": int(tp),
        },
        "cycle_detection": _cycle_detection_metrics(profiles[test], predicted_faults),
        "components": component_metrics,
    }
