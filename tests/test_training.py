import numpy as np

from hydraulic_faults.contract import CHANNELS, CONDITIONS, WINDOW_SIZE
from hydraulic_faults.model import window_features
from hydraulic_faults.train import _window_examples, split_cycles


def test_split_uses_disjoint_stable_cycles():
    profiles = np.tile([
        [100, 100, 0, 130, 0],
        [3, 73, 1, 90, 0],
        [20, 80, 2, 100, 0],
        [100, 90, 0, 115, 0],
        [100, 100, 0, 130, 1],
    ], (30, 1))
    train, test = split_cycles(profiles)
    assert len(train) == 96 and len(test) == 24
    assert not np.intersect1d(train, test).size
    assert (profiles[np.r_[train, test], 4] == 0).all()
    for column, labels in enumerate(CONDITIONS.values()):
        assert set(profiles[train, column]) == labels.keys()
        assert set(profiles[test, column]) == labels.keys()


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
