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

from hydraulic_faults.contract import CONDITIONS, NORMAL_CODES, SAMPLE_INTERVAL_MS, WINDOW_SIZE, InputError
from hydraulic_faults.dataset import MAX_ARCHIVE_BYTES, load_cycles
from hydraulic_faults.model import (
    MAX_TREE_DEPTH,
    VALVE_INDEX,
    VALVE_NEAR_FAILURE,
    Classifier,
    LinearHead,
    TreeHead,
    load_model,
    window_features,
)

SEED = 42
TEST_FRACTION = 0.2
ARCHIVE_GUARD_CYCLES = 10
ACCUMULATOR_INDEX = tuple(CONDITIONS).index("accumulator")
VALVE_ALPHA = 0.45


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
    codes, groups, counts = np.unique(profiles[faulty, :4], axis=0, return_inverse=True, return_counts=True)
    alerted_by_profile = np.bincount(groups[alerted[faulty]], minlength=len(counts))
    window_hits = np.bincount(groups, weights=windows_by_cycle[faulty].sum(axis=1))
    windows_per_cycle = windows_by_cycle.shape[1]
    profile_recall = window_hits / (counts * windows_per_cycle)
    healthy = windows_by_cycle[~faulty]
    # Reset at each cycle boundary, matching replay's reset marker.
    healthy_episodes = int(healthy[:, 0].sum() + (healthy[:, 1:] & ~healthy[:, :-1]).sum())
    healthy_seconds = healthy.size * SAMPLE_INTERVAL_MS / 1_000
    return {
        "fault_cycles": int(faulty.sum()),
        "fault_cycles_alerted": int(alerted[faulty].sum()),
        "healthy_cycles_alerted": int(alerted[~faulty].sum()),
        "fault_profiles": len(profile_recall),
        "fault_profiles_without_alert": int((profile_recall == 0).sum()),
        "macro_profile_window_recall": round(float(profile_recall.mean()), 4) if len(profile_recall) else None,
        "first_alert_cycle_second": _first_detection(windows_by_cycle[faulty]),
        "healthy_alarm_episodes": healthy_episodes,
        "healthy_monitoring_seconds": healthy_seconds,
        "healthy_alarm_episodes_per_hour": round(healthy_episodes * 3_600 / healthy_seconds, 4)
        if healthy_seconds else None,
        "fault_profile_breakdown": [
            {
                "conditions": dict(zip(CONDITIONS, map(int, profile), strict=True)),
                "cycles": int(count),
                "cycles_alerted": int(alerts),
                "cycles_missed": int(count - alerts),
                "windows_alerted": int(hits),
                "window_recall": round(float(recall), 4),
            }
            for profile, count, alerts, hits, recall in zip(
                codes, counts, alerted_by_profile, window_hits, profile_recall, strict=True,
            )
        ],
    }


def _first_detection(windows: np.ndarray) -> dict | None:
    detected = windows[windows.any(axis=1)]
    if not len(detected):
        return None
    first = detected.argmax(axis=1) + WINDOW_SIZE
    return {"median": float(np.median(first)), "p95": round(float(np.percentile(first, 95)), 2)}


def _critical_cycle_diagnosis(profiles: np.ndarray, predictions: np.ndarray) -> dict:
    by_cycle = predictions.reshape(len(profiles), -1, len(CONDITIONS))
    report = {}
    for column, (name, labels) in enumerate(CONDITIONS.items()):
        critical = "severe_leakage" if name == "pump" else "near_failure"
        critical_code = next(code for code, label in labels.items() if label == critical)
        relevant = profiles[:, column] == critical_code
        correct = by_cycle[relevant, :, column] == critical_code
        diagnosed = int(correct.any(axis=1).sum())
        report[name] = {
            "condition": critical,
            "cycles": len(correct),
            "cycles_correctly_diagnosed": diagnosed,
            "cycles_without_correct_diagnosis": len(correct) - diagnosed,
            "first_correct_diagnosis_cycle_second": _first_detection(correct),
        }
    return report


def _component_metrics(targets: np.ndarray, predictions: np.ndarray, raw: np.ndarray) -> dict:
    metrics = {}
    for column, (name, labels) in enumerate(CONDITIONS.items()):
        expected, guess, raw_guess = targets[:, column], predictions[:, column], raw[:, column]
        normal = NORMAL_CODES[column]
        codes = sorted(labels)
        diagnostics = {}
        for code in codes:
            if code == normal:
                continue
            actual = expected == code
            diagnostics[labels[code]] = {
                "windows": int(actual.sum()),
                "correct": int((actual & (guess == code)).sum()),
                "normal_before_gate": int((actual & (raw_guess == normal)).sum()),
                "additional_normal_after_gate": int((actual & (raw_guess != normal) & (guess == normal)).sum()),
                "correct_suppressed_by_gate": int((actual & (raw_guess == code) & (guess == normal)).sum()),
            }
        metrics[name] = {
            "label_order": [labels[code] for code in codes],
            "macro_f1": round(f1_score(expected, guess, average="macro"), 4),
            "confusion_matrix": confusion_matrix(expected, guess, labels=codes).tolist(),
            "diagnosis": diagnostics,
        }
    return metrics


def _window_examples(cycles: np.ndarray, profiles: np.ndarray):
    windows = np.lib.stride_tricks.sliding_window_view(cycles, WINDOW_SIZE, axis=1)
    features = window_features(windows.swapaxes(-1, -2))
    return features.reshape(-1, features.shape[-1]), np.repeat(profiles[:, :4], features.shape[1], axis=0)


def _unstable_cycle_check(model: Classifier, cycles: np.ndarray, profiles: np.ndarray, train: np.ndarray) -> dict:
    unstable = np.flatnonzero(profiles[:, 4] == 1)
    trained_profiles = set(map(tuple, profiles[train, :4]))
    seen = np.array([tuple(row) in trained_profiles for row in profiles[unstable, :4]], dtype=bool)
    checks = {}
    for name, indices in (("seen_profiles", unstable[seen]), ("unseen_profiles", unstable[~seen])):
        if not len(indices):
            checks[name] = {"cycles": 0}
            continue
        checks[name] = _evaluate_cycles(model, cycles[indices], profiles[indices])
    return checks


def _evaluate_cycles(model: Classifier, cycles: np.ndarray, profiles: np.ndarray) -> dict:
    features, targets = _window_examples(cycles, profiles)
    faults, predictions, raw = _cycle_predictions(model, features, len(profiles))
    return {
        "cycles": len(profiles),
        "healthy_cycles": int((profiles[:, :4] == NORMAL_CODES).all(axis=1).sum()),
        "cycle_detection": _cycle_detection_metrics(profiles, faults),
        "components": _component_metrics(targets, predictions, raw),
        "critical_cycle_diagnosis": _critical_cycle_diagnosis(profiles, predictions),
    }


def _cycle_predictions(
    model: Classifier, features: np.ndarray, cycle_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = features.reshape(cycle_count, -1, features.shape[-1])
    faults, conditions, raw = model.predict_cycles(windows)
    return faults.ravel(), conditions.reshape(-1, len(CONDITIONS)), raw.reshape(-1, len(CONDITIONS))


def _guarded_train_cycles(profiles: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    test = np.sort(test)
    near_test = np.searchsorted(test, train + ARCHIVE_GUARD_CYCLES, side="right") > np.searchsorted(
        test, train - ARCHIVE_GUARD_CYCLES,
    )
    # The only stable healthy run is already split for the same-run check.
    healthy = (profiles[train, :4] == NORMAL_CODES).all(axis=1)
    return train[~near_test | healthy]


def _archive_guard_check(cycles: np.ndarray, profiles: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    guarded = _guarded_train_cycles(profiles, train, test)
    report = {
        "guard_cycles": ARCHIVE_GUARD_CYCLES,
        "train_cycles": len(guarded),
        "removed_cycles": len(train) - len(guarded),
    }
    for column, (name, labels) in enumerate(CONDITIONS.items()):
        if set(profiles[guarded, column]) != labels.keys():
            return {**report, "skipped": f"guarded training data lacks some {name} conditions"}
    features, targets = _window_examples(cycles[guarded], profiles[guarded])
    try:
        model, _ = _fit_reference_model(features, targets)
    except ConvergenceWarning:
        return {**report, "skipped": "guarded training did not converge"}
    return {**report, "holdout": _evaluate_cycles(model, cycles[test], profiles[test])}


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


def _fit_reference_model(features: np.ndarray, targets: np.ndarray) -> tuple[Classifier, Classifier]:
    linear_model = fit_classifier(features, targets)
    tree = _fit_accumulator_tree((features - linear_model.mean) / linear_model.scale, targets[:, ACCUMULATOR_INDEX])
    heads = list(linear_model.heads)
    heads[ACCUMULATOR_INDEX] = tree
    return replace(linear_model, heads=tuple(heads), valve_alpha=VALVE_ALPHA), linear_model


def _save_model(classifier: Classifier, path: Path, features: np.ndarray, cycle_count: int, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, encoding="utf-8", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(classifier.to_dict(), output, allow_nan=False)
        exported = _cycle_predictions(load_model(temporary_path), features, cycle_count)
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
    model, linear_model = _fit_reference_model(train_x, train_y)
    predicted_faults, predicted, raw = _cycle_predictions(model, test_x, len(test))
    _, linear_predictions = linear_model.predict(test_x)
    baseline = np.column_stack([
        np.full(len(test_y), np.bincount(train_y[:, column]).argmax())
        for column in range(len(CONDITIONS))
    ])
    faults = (test_y != NORMAL_CODES).any(axis=1)
    tn, fp, fn, tp = confusion_matrix(faults, predicted_faults, labels=[False, True]).ravel()
    component_metrics = _component_metrics(test_y, predicted, raw)
    for column, (name, labels) in enumerate(CONDITIONS.items()):
        component_metrics[name]["majority_baseline_macro_f1"] = round(
            f1_score(test_y[:, column], baseline[:, column], labels=sorted(labels), average="macro", zero_division=0),
            4,
        )
    component_metrics["accumulator"]["linear_baseline_macro_f1"] = round(
        f1_score(test_y[:, ACCUMULATOR_INDEX], linear_predictions[:, ACCUMULATOR_INDEX], average="macro"), 4,
    )
    valve_targets = test_y[:, VALVE_INDEX]
    component_metrics["valve"]["single_window_baseline"] = {
        "macro_f1": round(f1_score(valve_targets, linear_predictions[:, VALVE_INDEX], average="macro"), 4),
        "near_failure_windows_correct": int(
            ((valve_targets == VALVE_NEAR_FAILURE) & (linear_predictions[:, VALVE_INDEX] == VALVE_NEAR_FAILURE)).sum(),
        ),
    }
    unstable_cycle_check = _unstable_cycle_check(model, cycles, profiles, train)
    archive_guard_check = _archive_guard_check(cycles, profiles, train, test)
    train_fault_cycles = (profiles[train, :4] != NORMAL_CODES).any(axis=1)
    report = {
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
        "valve_evidence_alpha": VALVE_ALPHA,
        "fault_detection": {
            "precision": round(precision_score(faults, predicted_faults, zero_division=0), 4),
            "recall": round(recall_score(faults, predicted_faults, zero_division=0), 4),
            "always_fault_baseline_precision": round(float(faults.mean()), 4),
            "true_negatives": int(tn), "false_positives": int(fp),
            "false_negatives": int(fn), "true_positives": int(tp),
        },
        "cycle_detection": _cycle_detection_metrics(profiles[test], predicted_faults),
        "components": component_metrics,
        "critical_cycle_diagnosis": _critical_cycle_diagnosis(profiles[test], predicted),
        "unstable_cycle_check": unstable_cycle_check,
        "archive_guard_check": archive_guard_check,
    }
    _save_model(model, model_path, test_x, len(test), (predicted_faults, predicted, raw))
    return report
