import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest

from hydraulic_faults.cli import detect, main
from hydraulic_faults.contract import CHANNELS, MAX_SAMPLE_BYTES, InputError
from hydraulic_faults.dataset import CYCLE_SECONDS, load_cycles


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


def test_loader_rejects_missing_cycle(tmp_path):
    path = tmp_path / "fixture.zip"
    _archive(path, truncate=True)
    with pytest.raises(InputError, match="cycle count"):
        load_cycles(path)


def test_detect_rejects_oversized_line_before_parsing(capsys):
    assert detect(None, BytesIO(b" " * (MAX_SAMPLE_BYTES + 1))) == 2
    assert "exceeds size limit" in capsys.readouterr().err
