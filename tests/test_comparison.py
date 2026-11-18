import numpy as np
import pytest
from test_valve import _capture

from hydraulic_faults.compare_valve import CANDIDATES, fit_candidate, grouped_intervals, selection_loss
from hydraulic_faults.release import encode
from hydraulic_faults.train_valve import augmentation_features
from hydraulic_faults.valve import FEATURE_COUNT, VALVE_LABELS, load_valve_model_bytes


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_candidate_export_keeps_predictions(candidate):
    rng = np.random.default_rng(17)
    targets = np.repeat(VALVE_LABELS, 30)
    features = rng.normal(size=(len(targets), FEATURE_COUNT)) + targets[:, None]
    fitted = fit_candidate(candidate, features, targets)
    loaded = load_valve_model_bytes(encode(fitted.to_dict()))
    np.testing.assert_array_equal(fitted.predict(features), loaded.predict(features))


def test_bootstrap_is_paired_and_selection_prioritizes_critical_misses():
    labels = np.repeat(VALVE_LABELS, 10)
    groups = np.arange(40) // 2
    result = grouped_intervals(labels, labels, labels, groups)
    assert result["accuracy_delta_95_percent"] == [0, 0]
    assert result["critical_recall_delta_95_percent"] == [0, 0]
    assert selection_loss(np.array([73, 100]), np.array([73, 73])) < selection_loss(
        np.array([73, 100]), np.array([100, 100]),
    )


def test_augmentation_is_reproducible_and_does_not_change_source():
    captures = np.stack([_capture(), _capture(65)])
    before = captures.copy()
    first, second = augmentation_features(captures), augmentation_features(captures)
    assert len(first) == 4
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
    np.testing.assert_array_equal(captures, before)
