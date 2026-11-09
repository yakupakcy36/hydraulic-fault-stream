import json
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from hydraulic_faults.cli import main
from hydraulic_faults.contract import InputError
from hydraulic_faults.model import LinearHead
from hydraulic_faults.valve import (
    BENCH_CAPTURE_START,
    CAPTURE_SAMPLES,
    FEATURE_COUNT,
    VALVE_LABELS,
    ValveModel,
    load_valve_model,
    transition_features,
)


def _capture(onset=55):
    inlet = np.full(CAPTURE_SAMPLES, 190.0)
    outlet = np.zeros(CAPTURE_SAMPLES)
    inlet[onset:] = 155.0
    outlet[onset:] = 120.0
    return np.stack((inlet, outlet))


def _model():
    head = LinearHead(
        np.array(VALVE_LABELS), np.zeros((len(VALVE_LABELS), FEATURE_COUNT)), np.array([1.0, 0, 0, 0]),
    )
    return ValveModel(np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT), head)


def test_transition_shape_is_relative_to_pressure_edge():
    original, first = transition_features(_capture(55))
    delayed, second = transition_features(_capture(70))
    np.testing.assert_array_equal(original, delayed)
    assert second - first == 15
    assert original.shape == (FEATURE_COUNT,)


def test_incomplete_or_unpaired_switch_is_unavailable():
    assert transition_features(_capture(120)) is None
    assert transition_features(np.zeros((2, CAPTURE_SAMPLES))) is None
    unpaired = _capture()
    unpaired[0] = 190
    assert transition_features(unpaired) is None


def test_valve_replay_and_detection_use_same_capture(tmp_path, monkeypatch, capsys):
    archive_path = tmp_path / "hydraulic.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("profile.txt", "100\t73\t0\t130\t0\n")
        for name, capture in zip(("PS1.txt", "PS2.txt"), _capture(), strict=True):
            full_cycle = np.zeros(6_000)
            full_cycle[BENCH_CAPTURE_START:BENCH_CAPTURE_START + CAPTURE_SAMPLES] = capture
            archive.writestr(name, "\t".join(map(str, full_cycle)) + "\n")
    assert main(["valve", "replay", str(archive_path), "--cycle", "0"]) == 0
    record = json.loads(capsys.readouterr().out)
    invalid = {**record, "ps2_bar": [0.0]}
    missing = {**record, "ps2_bar": [0.0] * CAPTURE_SAMPLES}
    records = "\n".join(map(json.dumps, (invalid, missing, record))).encode()
    monkeypatch.setattr("sys.stdin", SimpleNamespace(buffer=BytesIO(records)))
    path = tmp_path / "valve.json"
    path.write_text(json.dumps(_model().to_dict()), encoding="utf-8")
    assert main(["valve", "detect", "--model", str(path)]) == 0
    output = capsys.readouterr()
    unavailable, diagnosed = map(json.loads, output.out.splitlines())
    assert unavailable["status"] == "unavailable"
    assert diagnosed["status"] == "diagnosed" and diagnosed["condition"] == "near_failure"
    assert diagnosed["switch_offset_ms"] == 550
    assert "capture 1 ignored" in output.err


def test_valve_model_rejects_mismatched_feature_spec(tmp_path):
    path = tmp_path / "valve.json"
    payload = _model().to_dict()
    payload["feature_spec"] = "another-capture"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InputError, match="invalid valve model artifact"):
        load_valve_model(path)
