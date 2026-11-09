from itertools import product
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from hydraulic_faults.contract import CHANNELS, CONDITIONS, NORMAL_CODES, WINDOW_SIZE, InputError
from hydraulic_faults.model import FEATURE_COUNT, MAX_TREE_DEPTH, Classifier, window_features
from hydraulic_faults.train import (
    SEED,
    _component_metrics,
    _cycle_detection_metrics,
    _fit_accumulator_tree,
    _unstable_cycle_check,
    _window_examples,
    split_cycles,
)


def test_split_holds_out_entire_fault_profiles_and_healthy_tail():
    combinations = product(*(labels.keys() for labels in CONDITIONS.values()))
    profiles = np.tile([(*row, 0) for row in combinations] + [(*NORMAL_CODES, 1)], (10, 1))
    train, test = split_cycles(profiles)
    stable = np.flatnonzero(profiles[:, 4] == 0)
    healthy = (profiles[:, :4] == NORMAL_CODES).all(axis=1)
    np.testing.assert_array_equal(np.sort(np.r_[train, test]), stable)
    np.testing.assert_array_equal(test[healthy[test]], stable[healthy[stable]][-2:])
    train_faults = set(map(tuple, profiles[train[~healthy[train]], :4]))
    test_faults = set(map(tuple, profiles[test[~healthy[test]], :4]))
    assert len(train_faults) == 114 and len(test_faults) == 29
    assert train_faults.isdisjoint(test_faults)
    for column, labels in enumerate(CONDITIONS.values()):
        assert set(profiles[train, column]) == labels.keys()
        assert set(profiles[test, column]) == labels.keys()


def test_split_rejects_missing_healthy_cycles():
    profiles = np.array([[*NORMAL_CODES, 0], [20, 73, 1, 90, 0], [3, 80, 2, 100, 0]])
    with pytest.raises(InputError, match="two stable healthy cycles"):
        split_cycles(profiles)


def test_alarm_coverage_counts_cycles_and_weights_each_fault_profile_equally():
    profiles = np.array([[20, 73, 1, 90, 0]] * 2 + [[3, 80, 2, 100, 0], [*NORMAL_CODES, 0]])
    decisions = np.array([False, True, False, False, False, False, True, False])
    assert _cycle_detection_metrics(profiles, decisions) == {
        "fault_cycles": 3,
        "fault_cycles_alerted": 1,
        "healthy_cycles_alerted": 1,
        "fault_profiles": 2,
        "fault_profiles_without_alert": 1,
        "macro_profile_window_recall": 0.125,
        "first_alert_cycle_second": {"median": WINDOW_SIZE + 1, "p95": WINDOW_SIZE + 1},
    }


def test_diagnosis_distinguishes_normal_prediction_from_gate_suppression():
    targets = np.tile(NORMAL_CODES, (4, 1))
    raw = targets.copy()
    predicted = targets.copy()
    targets[:, -1] = 90
    raw[:, -1] = [130, 90, 100, 90]
    predicted[:, -1] = [130, 130, 130, 90]

    diagnosis = _component_metrics(targets, predicted, raw)["accumulator"]["diagnosis"]
    assert diagnosis["near_failure"] == {
        "windows": 4,
        "correct": 1,
        "normal_before_gate": 1,
        "additional_normal_after_gate": 2,
        "correct_suppressed_by_gate": 1,
    }


def test_unstable_check_separates_profiles_seen_in_training():
    profiles = np.array([
        [*NORMAL_CODES, 0],
        [*NORMAL_CODES, 1],
        [20, 73, 1, 90, 1],
    ])
    cycles = np.zeros((len(profiles), WINDOW_SIZE, len(CHANNELS)))
    model = Mock(spec=Classifier)
    conditions = np.array([NORMAL_CODES])
    model.predict_with_raw_conditions.return_value = (np.array([True]), conditions, conditions)
    result = _unstable_cycle_check(model, cycles, profiles, np.array([0]))

    seen = result["seen_profiles"]
    assert seen["cycles"] == 1 and seen["healthy_cycles"] == 1
    assert seen["cycle_detection"]["healthy_cycles_alerted"] == 1
    assert seen["cycle_detection"]["first_alert_cycle_second"] is None
    assert seen["cycle_detection"]["macro_profile_window_recall"] is None
    unseen = result["unseen_profiles"]
    assert unseen["cycles"] == 1 and unseen["healthy_cycles"] == 0
    assert unseen["cycle_detection"]["fault_cycles_alerted"] == 1


def test_accumulator_tree_matches_sklearn_at_split_boundaries():
    features = np.zeros((320, FEATURE_COUNT))
    features[:, 0] = np.linspace(-1.01, 1.02, len(features))
    labels = np.repeat([90, 100, 115, 130], 80)
    estimator = DecisionTreeClassifier(
        max_depth=MAX_TREE_DEPTH, min_samples_leaf=56, class_weight="balanced", random_state=SEED,
    ).fit(features, labels)
    tree = _fit_accumulator_tree(features, labels)
    thresholds = estimator.tree_.threshold[estimator.tree_.feature >= 0]
    assert thresholds.size > 0
    boundary_points = np.zeros((3 * len(thresholds), FEATURE_COUNT))
    nearest = thresholds.astype(np.float32)
    boundary_points[:, 0] = np.r_[
        np.nextafter(nearest, np.float32(-np.inf)), thresholds, np.nextafter(nearest, np.float32(np.inf)),
    ]
    candidates = np.vstack((features, boundary_points))
    np.testing.assert_array_equal(tree.predict(candidates), estimator.predict(candidates))


def test_offline_windows_match_online_features_without_crossing_cycle():
    n_samples = WINDOW_SIZE + 1
    n_channels = len(CHANNELS)
    cycles = np.arange(2 * n_samples * n_channels, dtype=float).reshape(2, n_samples, n_channels)
    profiles = np.array([[100, 100, 0, 130, 0], [20, 73, 1, 90, 0]])

    features, targets = _window_examples(cycles, profiles)
    assert len(features) == 4
    np.testing.assert_allclose(features[0], window_features(cycles[0, :-1]))
    np.testing.assert_allclose(features[2], window_features(cycles[1, :-1]))
    np.testing.assert_array_equal(targets, np.repeat(profiles[:, :4], 2, axis=0))
