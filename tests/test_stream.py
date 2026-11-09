import json
from dataclasses import replace

import numpy as np
import pytest

from hydraulic_faults.contract import CHANNELS, CONDITIONS, NORMAL_CODES, InputError, parse_sample
from hydraulic_faults.model import FEATURE_COUNT, Classifier, LinearHead, TreeHead, load_model
from hydraulic_faults.stream import DEFAULT_STALE_SECONDS, MAX_MACHINES, STALE_RETENTION_SECONDS, StreamDetector


def _model(fault=True):
    heads = []
    for name, conditions in CONDITIONS.items():
        labels = np.array(sorted(conditions))
        chosen = 1 if name == "pump" else next(iter(conditions))
        heads.append(LinearHead(
            labels, np.zeros((len(labels), FEATURE_COUNT)), (labels == chosen).astype(float),
        ))
    fault_head = LinearHead(
        np.array([0, 1]), np.zeros((2, FEATURE_COUNT)), np.array([not fault, fault], dtype=float),
    )
    return Classifier(np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT), fault_head, tuple(heads))


def _record(timestamp=0, machine="rig"):
    return {
        "machine_id": machine,
        "timestamp_ms": timestamp,
        "sensors": {channel.name: 1.0 for channel in CHANNELS},
    }


def _sample(timestamp, machine="rig"):
    return parse_sample(json.dumps(_record(timestamp, machine)).encode())


@pytest.mark.parametrize("value", [True, "5", float("nan"), float("inf"), 1e7])
def test_invalid_sensor_values(value):
    data = _record()
    data["sensors"]["ps1_bar"] = value
    with pytest.raises(InputError):
        parse_sample(json.dumps(data).encode())


def test_invalid_json_and_duplicate_fields():
    for raw in (b"\xff", b"[" * 2_049, b'{"machine_id":"x","machine_id":"y"}'):
        with pytest.raises(InputError):
            parse_sample(raw)


def test_state_isolates_machines_and_recovers_after_gap():
    detector = StreamDetector(_model())
    for tick in range(4):
        assert detector.ingest(_sample(tick * 1_000))["status"] == "warming_up"
    assert detector.ingest(_sample(4_000, "other"))["samples"] == 1

    result = detector.ingest(_sample(4_000))
    assert result["status"] == "fault"
    assert result["faults"] == ["pump"]
    assert result["conditions"]["pump"] == "weak_leakage"
    with pytest.raises(InputError, match="out-of-order"):
        detector.ingest(_sample(3_000))
    assert detector.ingest(_sample(5_000))["status"] == "fault"

    restarted = detector.ingest(_sample(7_000))
    assert restarted["status"] == "warming_up"
    assert restarted["gap"] and restarted["samples"] == 1
    for tick in range(8, 11):
        assert detector.ingest(_sample(tick * 1_000))["status"] == "warming_up"
    assert detector.ingest(_sample(11_000))["status"] == "fault"


def test_machine_capacity_is_bounded():
    detector = StreamDetector(_model())
    for index in range(MAX_MACHINES):
        detector.ingest(_sample(0, f"rig-{index}"))
    with pytest.raises(InputError, match="capacity"):
        detector.ingest(_sample(0, "overflow"))
    assert detector.ingest(_sample(1_000, "rig-0"))["samples"] == 2


def test_expired_machines_release_capacity():
    detector = StreamDetector(_model())
    for index in range(MAX_MACHINES):
        detector.ingest(_sample(0, f"rig-{index}"), now=0)

    assert len(detector.advance_time(DEFAULT_STALE_SECONDS)) == MAX_MACHINES
    with pytest.raises(InputError, match="capacity"):
        detector.ingest(_sample(0, "replacement"), now=DEFAULT_STALE_SECONDS)
    expiry = DEFAULT_STALE_SECONDS + STALE_RETENTION_SECONDS
    detector.advance_time(expiry)
    assert detector.ingest(_sample(0, "replacement"), now=expiry)["samples"] == 1


def test_idle_machine_emits_stale_once_and_restarts_warmup():
    detector = StreamDetector(_model())
    for tick in range(5):
        detector.ingest(_sample(tick * 1_000), now=tick)
    deadline = 4 + DEFAULT_STALE_SECONDS
    assert not detector.advance_time(deadline - 0.01)
    assert detector.advance_time(deadline) == [
        {"machine_id": "rig", "last_sample_timestamp_ms": 4_000, "status": "stale"},
    ]
    assert not detector.advance_time(deadline + 1)
    with pytest.raises(InputError, match="out-of-order"):
        detector.ingest(_sample(4_000), now=deadline + 1)
    result = detector.ingest(_sample(5_000), now=deadline + 1)
    assert result["status"] == "warming_up"
    assert result["samples"] == 1 and result["gap"]


def test_clock_reset_requires_an_explicit_boolean():
    detector = StreamDetector(_model())
    for tick in range(5):
        detector.ingest(_sample(tick * 1_000), now=tick)
    with pytest.raises(InputError, match="out-of-order"):
        detector.ingest(_sample(0), now=5)
    record = {**_record(0), "reset": 1}
    with pytest.raises(InputError, match="boolean"):
        parse_sample(json.dumps(record).encode())
    record["reset"] = True
    result = detector.ingest(parse_sample(json.dumps(record).encode()), now=5)
    assert result["status"] == "warming_up"
    assert result["samples"] == 1 and result["gap"]
    assert detector.ingest(_sample(1_000), now=6)["samples"] == 2


def test_normal_detection_reports_normal_component_states():
    model = _model(fault=False)
    faulty, codes = model.predict(np.zeros(FEATURE_COUNT))
    assert not faulty
    np.testing.assert_array_equal(codes, NORMAL_CODES)


def test_model_round_trip_and_bad_schema(tmp_path):
    model = _model()
    path = tmp_path / "model.json"
    payload = model.to_dict()
    assert payload["schema_version"] == 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    features = np.arange(FEATURE_COUNT, dtype=float)
    actual_fault, actual_codes = load_model(path).predict(features)
    expected_fault, expected_codes = model.predict(features)
    assert actual_fault == expected_fault
    np.testing.assert_array_equal(actual_codes, expected_codes)

    payload["scale"][0] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InputError):
        load_model(path)

    path.write_bytes(b" " * 65_537)
    with pytest.raises(InputError, match="size limit"):
        load_model(path)


def test_tree_artifact_round_trip_and_invalid_children(tmp_path):
    model = _model()
    tree = TreeHead(
        np.array([90, 100, 115, 130]),
        np.array([[1, 2], [-1, -1], [-1, -1]]),
        np.array([0, -2, -2]),
        np.array([0.5, -2, -2]),
        np.array([0, 0, 3]),
    )
    model = replace(model, heads=(*model.heads[:-1], tree))
    payload = model.to_dict()
    assert payload["schema_version"] == 3
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    features = np.zeros((2, FEATURE_COUNT))
    features[:, 0] = [0.5, 0.6]
    loaded = load_model(path)
    np.testing.assert_array_equal(loaded.predict(features)[1][:, -1], [90, 130])
    assert loaded.predict(features[0])[1][-1] == 90

    for children in ([0, 2], [1, 1]):
        payload["heads"]["accumulator"]["tree"]["children"][0] = children
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InputError, match="invalid model artifact"):
            load_model(path)
