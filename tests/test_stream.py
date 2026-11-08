import json

import numpy as np
import pytest

from hydraulic_faults.contract import CHANNELS, CONDITIONS, InputError, parse_sample
from hydraulic_faults.model import FEATURE_COUNT, Classifier, LinearHead, load_model
from hydraulic_faults.stream import MAX_MACHINES, StreamDetector


def _model():
    heads = []
    for name, conditions in CONDITIONS.items():
        labels = np.array(sorted(conditions))
        chosen = 1 if name == "pump" else next(iter(conditions))
        heads.append(LinearHead(
            labels, np.zeros((len(labels), FEATURE_COUNT)), (labels == chosen).astype(float),
        ))
    return Classifier(np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT), tuple(heads))


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


def test_model_round_trip_and_bad_schema(tmp_path):
    model = _model()
    path = tmp_path / "model.json"
    payload = model.to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    features = np.arange(FEATURE_COUNT, dtype=float)
    np.testing.assert_array_equal(load_model(path).predict(features), model.predict(features))

    payload["scale"][0] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InputError):
        load_model(path)

    path.write_bytes(b" " * 65_537)
    with pytest.raises(InputError, match="size limit"):
        load_model(path)
