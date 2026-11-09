import json
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from hydraulic_faults.cli import main
from hydraulic_faults.contract import InputError
from hydraulic_faults.model import LinearHead
from hydraulic_faults.train_valve import SKEW_CONTEXT_SAMPLES, _relative_skew_check
from hydraulic_faults.valve import (
    BENCH_CAPTURE_START,
    CAPTURE_SAMPLES,
    FEATURE_COUNT,
    VALVE_LABELS,
    ValveModel,
    load_valve_captures,
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


def test_capture_loader_can_read_adjacent_samples_without_changing_replay_window(tmp_path):
    archive_path = tmp_path / "hydraulic.zip"
    row = np.arange(6_000)
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("profile.txt", "100\t73\t0\t130\t0\n")
        for name in ("PS1.txt", "PS2.txt"):
            archive.writestr(name, "\t".join(map(str, row)) + "\n")
    context, _ = load_valve_captures(archive_path, context_samples=2)
    normal, _ = load_valve_captures(archive_path, cycle=0)
    np.testing.assert_array_equal(normal, context[:, :, 2:-2])
    np.testing.assert_array_equal(context[0, 0, :2], [BENCH_CAPTURE_START - 2, BENCH_CAPTURE_START - 1])
    np.testing.assert_array_equal(
        context[0, 1, -2:], [BENCH_CAPTURE_START + CAPTURE_SAMPLES, BENCH_CAPTURE_START + CAPTURE_SAMPLES + 1],
    )


def test_skew_check_shifts_only_ps1_and_uses_real_context(monkeypatch):
    trace = np.arange(CAPTURE_SAMPLES + 2 * SKEW_CONTEXT_SAMPLES)
    context = np.stack((trace, trace + 1_000))[None, :, :]
    original = context.copy()
    seen = []

    def extract(capture):
        seen.append(capture.copy())
        return np.zeros(FEATURE_COUNT), 0

    monkeypatch.setattr("hydraulic_faults.train_valve.transition_features", extract)
    results = _relative_skew_check(_model(), np.array([73]), context)
    assert [row["ps1_source_offset_ms"] for row in results] == [-20, -10, 10, 20]
    for offset, capture, result in zip((-2, -1, 1, 2), seen, results, strict=True):
        start = SKEW_CONTEXT_SAMPLES
        np.testing.assert_array_equal(capture[0], trace[start + offset:start + offset + CAPTURE_SAMPLES])
        np.testing.assert_array_equal(capture[1], original[0, 1, start:start + CAPTURE_SAMPLES])
        assert result["critical_correct"] == 1
    np.testing.assert_array_equal(context, original)


def test_valve_model_rejects_mismatched_feature_spec(tmp_path):
    path = tmp_path / "valve.json"
    payload = _model().to_dict()
    payload["feature_spec"] = "another-capture"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InputError, match="invalid valve model artifact"):
        load_valve_model(path)
