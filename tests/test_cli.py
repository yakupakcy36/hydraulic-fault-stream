import json
import subprocess
import sys
from io import BytesIO
from threading import Event
from zipfile import ZipFile

import numpy as np
import pytest

from hydraulic_faults.cli import detect, main
from hydraulic_faults.contract import CHANNELS, MAX_SAMPLE_BYTES, InputError
from hydraulic_faults.dataset import CYCLE_SECONDS, load_cycles
from hydraulic_faults.stream import StreamDetector


def _archive(path, truncate=False):
    with ZipFile(path, "w") as archive:
        archive.writestr("profile.txt", "100\t100\t0\t130\t0\n3\t73\t2\t90\t1\n")
        for index, channel in enumerate(CHANNELS):
            sample_count = CYCLE_SECONDS * channel.hz
            rows = [
                "\t".join(str(value + index) for value in range(sample_count)),
                "\t".join(["12"] * sample_count),
            ]
            if truncate and index == 0:
                rows.pop()
            archive.writestr(channel.member, "\n".join(rows) + "\n")


def test_loader_aligns_sampling_rates_and_replay(tmp_path, capsys):
    path = tmp_path / "fixture.zip"
    _archive(path)
    values, profiles = load_cycles(path)
    assert values.shape == (2, CYCLE_SECONDS, len(CHANNELS))
    assert values[0, 0, 0] == 49.5
    assert values[0, 1, 0] == 149.5
    assert values[0, 0, 4] == 8.5
    assert profiles[1].tolist() == [3, 73, 2, 90, 1]
    single, profile = load_cycles(path, cycle=1)
    np.testing.assert_array_equal(single[0], values[1])
    np.testing.assert_array_equal(profile[0], profiles[1])

    assert main(["replay", str(path), "--cycle", "1", "--seconds", "6", "--speed", "0"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 6
    assert records[1]["timestamp_ms"] - records[0]["timestamp_ms"] == 1_000
    assert records[0]["sensors"] == {channel.name: 12.0 for channel in CHANNELS}
    assert records[0]["reset"] is True
    assert all("reset" not in record for record in records[1:])


def test_loader_rejects_missing_cycle(tmp_path):
    path = tmp_path / "fixture.zip"
    _archive(path, truncate=True)
    with pytest.raises(InputError, match="cycle count"):
        load_cycles(path)


def test_replay_exits_cleanly_when_output_pipe_closes(tmp_path):
    path = tmp_path / "fixture.zip"
    _archive(path)
    command = [
        sys.executable, "-m", "hydraulic_faults.cli",
        "replay", str(path), "--cycle", "0", "--speed", "100",
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as replay:
        try:
            replay.stdout.close()
            assert replay.wait(timeout=20) == 1
            assert not replay.stderr.read()
        finally:
            if replay.poll() is None:
                replay.kill()


def test_detect_rejects_oversized_line_before_parsing(capsys):
    assert detect(StreamDetector(None), BytesIO(b" " * (MAX_SAMPLE_BYTES + 1))) == 2
    assert "exceeds size limit" in capsys.readouterr().err


def _valid_line():
    return json.dumps({
        "machine_id": "rig",
        "timestamp_ms": 1_000,
        "sensors": {channel.name: 1.0 for channel in CHANNELS},
    }).encode()


def test_invalid_only_stream_fails_and_mixed_stream_recovers(capsys):
    assert detect(StreamDetector(None), BytesIO(b"bad\nbad\n")) == 2
    output = capsys.readouterr()
    assert not output.out and output.err.count("ignored") == 2

    assert detect(StreamDetector(None), BytesIO(b"bad\n" + _valid_line())) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "warming_up"
    assert output.err.count("ignored") == 1
    assert detect(StreamDetector(None), BytesIO()) == 0


def test_detect_emits_stale_while_the_input_is_open(monkeypatch):
    stale_emitted = Event()
    pending = BytesIO(_valid_line() + b"\n")

    class PausedSource:
        def read(self, size):
            return pending.read(size) or self._wait_for_stale()

        def _wait_for_stale(self):
            if not stale_emitted.wait(timeout=3):
                raise OSError("stale event was not emitted before EOF")
            return b""

    events = []

    def collect(event):
        events.append(event)
        if event["status"] == "stale":
            stale_emitted.set()

    monkeypatch.setattr("hydraulic_faults.cli._print_record", collect)
    assert detect(StreamDetector(None, stale_after_seconds=0.1), PausedSource()) == 0
    assert [event["status"] for event in events] == ["warming_up", "stale"]
