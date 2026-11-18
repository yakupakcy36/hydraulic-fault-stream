import json
from dataclasses import replace

import numpy as np
import pytest

from hydraulic_faults.contract import CHANNELS, CONDITIONS, NORMAL_CODES, InputError, parse_sample
from hydraulic_faults.model import FEATURE_COUNT, VALVE_INDEX, Classifier, LinearHead, TreeHead, load_model
from hydraulic_faults.quality import SensorRule
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


def _temporal_model():
    model = _model()
    heads = list(model.heads)
    valve = heads[VALVE_INDEX]
    coefficients = valve.coefficients.copy()
    coefficients[0, 0] = 1
    heads[VALVE_INDEX] = replace(valve, coefficients=coefficients, intercept=np.array([0.0, -5, -5, 0]))
    return replace(model, heads=tuple(heads), valve_alpha=0.45)


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


@pytest.mark.parametrize("defect", ["duplicate", "version_type", "window_type"])
def test_model_rejects_ambiguous_metadata(tmp_path, defect):
    data = _model().to_dict()
    if defect == "version_type":
        data["schema_version"] = float(data["schema_version"])
    elif defect == "window_type":
        data["window_size"] = float(data["window_size"])
    raw = json.dumps(data)
    if defect == "duplicate":
        raw = raw[:-1] + ',"schema_version":2}'
    path = tmp_path / "ambiguous.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(InputError, match="invalid model artifact"):
        load_model(path)


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


def test_flatline_suppresses_fault_inference_and_warms_up_after_recovery():
    detector = StreamDetector(_model(), quality_rules={"rig": (SensorRule(0, None, 5),)})
    for tick in range(4):
        assert detector.ingest(_sample(tick * 1_000), now=tick)["status"] == "warming_up"
    assert detector.ingest(_sample(4_000), now=4) == {
        "machine_id": "rig", "timestamp_ms": 4_000, "status": "data_quality", "issues": {"ps1_bar": "flatline"},
    }
    assert detector.ingest(_sample(5_000), now=5)["status"] == "data_quality"
    for tick in range(6, 11):
        sample = _sample(tick * 1_000)
        good = replace(sample, values=(float(tick), *sample.values[1:]))
        result = detector.ingest(good, now=tick)
        assert result["status"] == ("fault" if tick == 10 else "warming_up")


def test_quality_range_is_machine_specific_and_gap_resets_flatline_history():
    detector = StreamDetector(_model(), quality_rules={"rig": (SensorRule(0, (1.0, 2.0), 2),)})
    sample = _sample(0)
    assert detector.ingest(replace(sample, values=(3.0, *sample.values[1:])), now=0)["issues"] == {
        "ps1_bar": "out_of_range",
    }
    assert detector.ingest(_sample(1_000), now=1)["samples"] == 1
    assert detector.ingest(_sample(2_000), now=2)["issues"] == {"ps1_bar": "flatline"}
    assert detector.ingest(_sample(4_000), now=4)["samples"] == 1
    for tick in range(5, 10):
        result = detector.ingest(_sample(tick * 1_000, "other"), now=tick)
    assert result["status"] == "fault"


def test_normal_detection_reports_normal_component_states():
    model = _model(fault=False)
    features = np.zeros(FEATURE_COUNT)
    faulty, codes = model.predict(features)
    assert not faulty
    np.testing.assert_array_equal(codes, NORMAL_CODES)
    _, reported, raw = model.predict_with_raw_conditions(features)
    np.testing.assert_array_equal(reported, codes)
    assert raw[2] == 1


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


def test_temporal_model_round_trip_preserves_immediate_critical_valve_evidence(tmp_path):
    model = _temporal_model()
    payload = model.to_dict()
    assert payload["schema_version"] == 4
    path = tmp_path / "temporal.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    model = load_model(path)
    features = np.zeros((2, 2, FEATURE_COUNT))
    features[:, :, 0] = [[10, -1], [-10, 1]]
    _, gated, raw = model.predict_cycles(features)
    np.testing.assert_array_equal(gated[:, :, VALVE_INDEX], [[73, 73], [100, 73]])
    np.testing.assert_array_equal(gated, raw)

    payload["valve_alpha"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InputError, match="invalid model artifact"):
        load_model(path)


def test_stream_restarts_valve_evidence_after_quality_rejection():
    detector = StreamDetector(_temporal_model(), quality_rules={"rig": (SensorRule(0, (-30, 30), None),)})

    def feed(tick, ps1):
        sample = _sample(tick * 1_000)
        return detector.ingest(replace(sample, values=(ps1, *sample.values[1:])), now=tick)

    for tick in range(5):
        result = feed(tick, 20)
    assert result["conditions"]["valve"] == "near_failure"
    assert detector.machines["rig"].valve_scores is not None
    assert feed(5, 40)["status"] == "data_quality"
    assert detector.machines["rig"].valve_scores is None
    for tick in range(6, 11):
        result = feed(tick, -1)
    assert result["conditions"]["valve"] == "normal"


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
