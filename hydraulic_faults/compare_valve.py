"""Fixed candidates, grouped nested selection and paired outer-fold evidence."""

import hashlib
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from hydraulic_faults.contract import InputError
from hydraulic_faults.model import LinearHead
from hydraulic_faults.release import atomic_json
from hydraulic_faults.train_valve import _extract_features, _measure, augmentation_features
from hydraulic_faults.valve import VALVE_LABELS, ValveModel, load_valve_captures

CANDIDATES = ("lda", "shrinkage_lda", "logistic", "augmented_lda")
SEED = 731


def fit_candidate(name: str, features: np.ndarray, targets: np.ndarray) -> ValveModel:
    if set(targets) != set(VALVE_LABELS):
        raise InputError("comparison fold lacks a valve severity")
    scaler = StandardScaler().fit(features)
    estimators = {
        "lda": LinearDiscriminantAnalysis(),
        "augmented_lda": LinearDiscriminantAnalysis(),
        "shrinkage_lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "logistic": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED),
    }
    estimator = estimators[name].fit(scaler.transform(features), targets)
    return ValveModel(scaler.mean_, scaler.scale_, LinearHead(estimator.classes_, estimator.coef_, estimator.intercept_))


def selection_loss(targets, predicted) -> tuple[int, int, int]:
    """Predeclared lexicographic objective: missed critical, false critical, all errors."""
    return (int(((targets == 73) & (predicted != 73)).sum()),
            int(((targets != 73) & (predicted == 73)).sum()), int((targets != predicted).sum()))


def grouped_intervals(targets, baseline, selected, groups) -> dict:
    """Paired group bootstrap, not an assumption of independent overlapping windows."""
    unique = np.unique(groups)
    counts = np.array([
        [np.sum(groups == group), np.sum((groups == group) & (targets == 73)),
         np.sum((groups == group) & (selected == targets)) - np.sum((groups == group) & (baseline == targets)),
         np.sum((groups == group) & (targets == 73) & (selected == 73))
         - np.sum((groups == group) & (targets == 73) & (baseline == 73))]
        for group in unique
    ])
    rng = np.random.default_rng(SEED)
    samples = counts[rng.integers(0, len(unique), size=(1000, len(unique)))].sum(axis=1)
    return {
        "accuracy_delta_95_percent": np.quantile(samples[:, 2] / samples[:, 0], [0.025, 0.975]).tolist(),
        "critical_recall_delta_95_percent": np.quantile(
            samples[samples[:, 1] > 0, 3] / samples[samples[:, 1] > 0, 1], [0.025, 0.975],
        ).tolist(),
        "unit": "complete fault-profile group", "replicates": 1000,
    }


def compare_valve(archive: Path, output: Path, objective: str = "clean") -> dict:
    if objective not in ("clean", "robust"):
        raise InputError("unknown comparison objective")
    if output.exists() or output.resolve() == archive.resolve():
        raise InputError("comparison report requires a new output path")
    captures, profiles = load_valve_captures(archive)
    stable = np.flatnonzero(profiles[:, 4] == 0)
    captures, targets = captures[stable], profiles[stable, 1]
    features, available = _extract_features(captures)
    augmented = augmentation_features(captures)

    def fit(name, indices):
        variants = [(features, available)] + (augmented if name == "augmented_lda" else [])
        xs, ys = [], []
        for values, usable in variants:
            selected_indices = indices[usable[indices]]
            xs.append(values[selected_indices])
            ys.append(targets[selected_indices])
        return fit_candidate(name, np.concatenate(xs), np.concatenate(ys))
    _, groups = np.unique(profiles[stable, :4], axis=0, return_inverse=True)
    baseline = np.full(len(targets), -1)
    selected = baseline.copy()
    stress_predictions = {name: {"baseline": baseline.copy(), "selected": baseline.copy()}
                          for name in ("gain_plus_2_percent", "offset_plus_2_bar", "noise_std_0_5_bar", "clipped_at_170_bar")}
    rng = np.random.default_rng(SEED)
    stress_captures = {
        "gain_plus_2_percent": captures * 1.02, "offset_plus_2_bar": captures + 2,
        "noise_std_0_5_bar": captures + rng.normal(0, 0.5, captures.shape),
        "clipped_at_170_bar": np.minimum(captures, 170),
    }
    stress_features = {name: _extract_features(values) for name, values in stress_captures.items()}
    folds = []
    outer = GroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    for train, test in outer.split(features, targets, groups):
        losses = {}
        inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=SEED + 1).split(
            features[train], targets[train], groups[train],
        ))
        for name in CANDIDATES:
            prediction = np.full(len(train), -1)
            stress_inner = {stress: prediction.copy() for stress in ("gain_plus_2_percent", "noise_std_0_5_bar")}
            for fit_indices, validate in inner:
                model = fit(name, train[fit_indices])
                accepted = validate[available[train[validate]]]
                prediction[accepted] = model.predict(features[train[accepted]])
                if objective == "robust":
                    for stress, guesses in stress_inner.items():
                        values, usable = stress_features[stress]
                        accepted_stress = validate[usable[train[validate]]]
                        guesses[accepted_stress] = model.predict(values[train[accepted_stress]])
            predictions = [prediction, *stress_inner.values()] if objective == "robust" else [prediction]
            losses[name] = selection_loss(np.tile(targets[train], len(predictions)), np.concatenate(predictions))
        chosen = min(CANDIDATES, key=lambda name: losses[name])
        models = {"baseline": fit("lda", train), "selected": fit(chosen, train)}
        accepted = test[available[test]]
        baseline[accepted] = models["baseline"].predict(features[accepted])
        selected[accepted] = models["selected"].predict(features[accepted])
        for name, (values, usable) in stress_features.items():
            accepted_stress = test[usable[test]]
            for policy, model in models.items():
                stress_predictions[name][policy][accepted_stress] = model.predict(values[accepted_stress])
        folds.append({
            "train_archive_indices": stable[train].tolist(), "test_archive_indices": stable[test].tolist(),
            "selected_candidate": chosen, "inner_losses": losses,
            "baseline": _measure(targets[test], baseline[test]), "selected": _measure(targets[test], selected[test]),
        })
    with archive.open("rb") as data:
        dataset_hash = hashlib.file_digest(data, "sha256").hexdigest()
    report = {
        "protocol": "nested_grouped_valve_v3", "objective": objective, "seed": SEED, "dataset_sha256": dataset_hash,
        "selection_scenarios": ["clean", "gain_plus_2_percent", "noise_std_0_5_bar"] if objective == "robust" else ["clean"],
        "augmentation": {"seed": 732, "gain_factors": [0.98, 1.02], "noise_std_bar": 0.5, "noise_replicates": 2},
        "candidates": list(CANDIDATES), "selection_objective": ["missed_critical", "false_critical", "total_errors"],
        "scope": "development evidence; archive previously studied; not external validation",
        "healthy_run_policy": "all stable healthy cycles are one group; no independent healthy run",
        "baseline": _measure(targets, baseline), "selected": _measure(targets, selected), "folds": folds,
        "paired_group_bootstrap": grouped_intervals(targets, baseline, selected, groups),
        "synthetic_stress": {name: {policy: _measure(targets, predicted) for policy, predicted in policies.items()}
                             for name, policies in stress_predictions.items()},
        "deployment_decision": "comparison only; no automatic promotion or alarm policy change",
    }
    atomic_json(output, report)
    return report
