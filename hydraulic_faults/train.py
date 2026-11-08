import hashlib
import json
import os
import platform
import tempfile
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hydraulic_faults.contract import CONDITIONS, NORMAL_CODES, WINDOW_SIZE, InputError
from hydraulic_faults.dataset import MAX_ARCHIVE_BYTES, load_cycles
from hydraulic_faults.model import Classifier, LinearHead, load_model, window_features

SEED = 42
TEST_FRACTION = 0.2


def split_cycles(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stable = np.flatnonzero(profiles[:, 4] == 0)
    fault = (profiles[stable, :4] != NORMAL_CODES).any(axis=1)
    train, test = train_test_split(
        stable, test_size=TEST_FRACTION, random_state=SEED, stratify=fault,
    )
    for column, (name, labels) in enumerate(CONDITIONS.items()):
        if set(profiles[train, column]) != labels.keys() or set(profiles[test, column]) != labels.keys():
            raise InputError(f"train and test must each cover all {name} conditions")
    return train, test


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
    model = fit_classifier(train_x, train_y)
    predicted_faults, predicted = model.predict(test_x)
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
    _save_model(model, model_path, test_x, (predicted_faults, predicted))
    return {
        "dataset_sha256": archive_sha,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "seed": SEED,
        "train_cycles": len(train),
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
        "components": component_metrics,
    }
