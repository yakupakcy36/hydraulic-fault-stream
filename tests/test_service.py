import json
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from test_stream import _model, _record
from test_valve import _model as valve_model
from test_valve import _timed_record

from hydraulic_faults.contract import CHANNELS, InputError
from hydraulic_faults.release import activate_release, atomic_json, load_active, load_release, make_release
from hydraulic_faults.service import ServiceEngine, run_service, validate_service_config


def config(kind="rolling"):
    channels = [channel.name for channel in CHANNELS] if kind == "rolling" else ["ps1_bar", "ps2_bar"]
    options = {"range": [-10, 1000]}
    if kind == "valve":
        options["saturation_values"] = [-10, 1000]
    return {
        "schema_version": 1, "machines": {"rig": {channel: dict(options) for channel in channels}},
        "max_age_seconds": 60, "stale_after_seconds": 0.1,
        "output_timeout_seconds": 0.2, "status_interval_seconds": 0.1,
    }


def bundle(kind="rolling", marker=0):
    model = _model() if kind == "rolling" else valve_model()
    return make_release(kind, model.to_dict(), {"marker": marker}, {"test_fixture": True}), model


def activated(tmp_path, kind="rolling"):
    release = tmp_path / "release.json"
    settings = tmp_path / "config.json"
    pointer = tmp_path / "active.json"
    atomic_json(release, bundle(kind)[0])
    atomic_json(settings, config(kind))
    activate_release(release, pointer, settings)
    return release, settings, pointer


def test_release_corruption_and_atomic_activation_rollback(tmp_path):
    first, settings, pointer = activated(tmp_path)
    previous = pointer.read_bytes()
    second = tmp_path / "second.json"
    candidate = bundle(marker=1)[0]
    candidate["payload"]["report"]["marker"] = 2
    atomic_json(second, candidate)
    with pytest.raises(InputError, match="hash mismatch"):
        activate_release(second, pointer, settings)
    assert pointer.read_bytes() == previous
    second.unlink()
    atomic_json(second, bundle(marker=1)[0])
    activate_release(second, pointer, settings)
    assert load_active(pointer)[0]["payload"]["report"]["marker"] == 1
    activate_release(first, pointer, settings)
    assert pointer.read_bytes() == previous
    with pytest.raises(FileExistsError):
        atomic_json(first, bundle(marker=9)[0])
    assert load_release(first)[0]["payload"]["report"]["marker"] == 0


def test_activation_snapshots_config_and_checks_identity(tmp_path):
    _, settings, pointer = activated(tmp_path)
    settings.write_text('{}')
    assert load_active(pointer)[2] == config()
    modified = json.loads(pointer.read_bytes())
    modified["config"]["max_age_seconds"] = 2
    pointer.write_text(json.dumps(modified))
    with pytest.raises(InputError, match="active pointer"):
        load_active(pointer)


@pytest.mark.parametrize("mutation", ["missing_range", "unknown_channel", "empty", "nan", "extra", "boolean"])
def test_strict_configuration_rejects_incomplete_or_ambiguous_settings(mutation):
    data = config()
    if mutation == "missing_range":
        data["machines"]["rig"]["ps1_bar"] = {"flatline_samples": 3}
    elif mutation == "unknown_channel":
        data["machines"]["rig"]["unknown"] = {"range": [0, 1]}
    elif mutation == "empty":
        data["machines"] = {}
    elif mutation == "nan":
        data["max_age_seconds"] = float("nan")
    elif mutation == "extra":
        data["allow_all"] = True
    else:
        data["schema_version"] = True
    with pytest.raises(InputError):
        validate_service_config(data, "rolling")


def test_strict_engine_unknown_machine_cannot_allocate_state_and_quality_is_enforced():
    release, model = bundle()
    settings = config()
    settings["machines"]["rig"]["ps1_bar"]["range"] = [0, 0.5]
    engine = ServiceEngine(release, model, settings)
    unknown = engine.ingest(json.dumps(_record(1000, "typo")).encode(), now=0, now_ms=1000)
    assert unknown["reason"] == "unknown_machine"
    assert not engine.detector.machines
    result = engine.ingest(json.dumps(_record(1000)).encode(), now=0, now_ms=1000)
    assert result["result"]["status"] == "data_quality"
    assert engine.health(1)["counts"]["rejected"] == 1


def test_valve_invalid_range_does_not_poison_valid_retry():
    release, model = bundle("valve")
    engine = ServiceEngine(release, model, config("valve"))
    invalid = _timed_record()
    invalid["ps1_bar"][0] = 2000
    result = engine.ingest(json.dumps(invalid).encode(), now=0, now_ms=10_000)
    assert result["reason"] == "sensor_range"
    assert not engine.detector.last_timestamp
    result = engine.ingest(json.dumps(_timed_record()).encode(), now=0, now_ms=10_000)
    assert result["result"]["status"] == "diagnosed"


def test_valve_clipping_is_rejected_at_configured_rail():
    release, model = bundle("valve")
    settings = config("valve")
    settings["machines"]["rig"]["ps1_bar"]["saturation_values"] = [170]
    engine = ServiceEngine(release, model, settings)
    clipped = _timed_record()
    clipped["ps1_bar"] = [min(value, 170) for value in clipped["ps1_bar"]]
    result = engine.ingest(json.dumps(clipped).encode(), now=0, now_ms=10_000)
    assert result["reason"] == "sensor_saturation"
    assert not engine.detector.last_timestamp


def test_service_eof_and_rate_limited_rejections_keep_exact_counts():
    release, model = bundle()
    engine = ServiceEngine(release, model, config())
    events = []

    class Writer:
        def write(self, event):
            events.append(event)

    assert run_service(engine, BytesIO(b"bad\n" * 100), Writer()) == 4
    assert events[0]["event_type"] == "started"
    assert events[-1]["event_type"] == "ended"
    assert events[-2]["counts"] == {"received": 100, "rejected": 100}
    assert sum(event["event_type"] == "rejected" for event in events) < 100
    assert all(event["release_id"] == release["release_id"] for event in events)


def test_service_real_blocked_stdout_exits_with_deadline(tmp_path):
    _, _, pointer = activated(tmp_path)
    record = {**_record(time.time_ns() // 1_000_000), "reset": True}
    source = tmp_path / "input.ndjson"
    source.write_bytes((json.dumps(record).encode() + b"\n") * 10_000)
    with source.open("rb") as incoming, subprocess.Popen(
        [sys.executable, "-m", "hydraulic_faults.cli", "serve", "--active", str(pointer)],
        stdin=incoming, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) as process:
        try:
            assert process.wait(timeout=15) == 3
            assert not process.stderr.read()
        finally:
            if process.poll() is None:
                process.kill()


def test_service_cli_eof_is_monitoring_loss(tmp_path):
    _, _, pointer = activated(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "hydraulic_faults.cli", "serve", "--active", str(pointer)],
        input=b"", capture_output=True, timeout=15,
    )
    assert result.returncode == 4
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert [event["event_type"] for event in events] == ["started", "health", "ended"]
    assert not result.stderr


def test_closed_output_pipe_has_distinct_exit(tmp_path):
    _, _, pointer = activated(tmp_path)
    with subprocess.Popen(
        [sys.executable, "-m", "hydraulic_faults.cli", "serve", "--active", str(pointer)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) as process:
        try:
            process.stdout.close()
            assert process.wait(timeout=15) == 1
            assert not process.stderr.read()
        finally:
            if process.poll() is None:
                process.kill()


def test_activation_cannot_replace_an_unrelated_file(tmp_path):
    release, settings, _ = activated(tmp_path)
    other = tmp_path / "important.json"
    other.write_text('{"unrelated":true}')
    with pytest.raises(InputError):
        activate_release(release, other, settings)
    assert other.read_text() == '{"unrelated":true}'


def test_failed_training_never_publishes_or_changes_active_release(tmp_path, monkeypatch):
    import hydraulic_faults.release as releases

    _, _, pointer = activated(tmp_path)
    original = pointer.read_bytes()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w"):
        pass
    monkeypatch.setattr("hydraulic_faults.dataset._profiles", lambda _: None)
    monkeypatch.setattr("hydraulic_faults.train.split_cycles", lambda _: (np.array([0]), np.array([1])))
    monkeypatch.setattr(releases, "source_provenance", lambda _: {"source_id": "fixture", "git_revision": "fixture"})

    def fail(*args):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr("hydraulic_faults.train_valve.train_valve", fail)
    before = set(tmp_path.iterdir())
    output = tmp_path / "failed-release.json"
    with pytest.raises(RuntimeError, match="evaluation failed"):
        releases.train_release(archive, output, "valve", Path(releases.__file__).resolve().parents[1])
    assert not output.exists()
    assert set(tmp_path.iterdir()) == before
    assert pointer.read_bytes() == original
