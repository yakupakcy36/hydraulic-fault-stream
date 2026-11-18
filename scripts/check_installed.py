"""Run with the installed wheel's Python -I from outside the repository."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import hydraulic_faults
from hydraulic_faults.contract import CHANNELS, CONDITIONS
from hydraulic_faults.model import FEATURE_COUNT, Classifier, LinearHead
from hydraulic_faults.release import activate_release, atomic_json, make_release
from hydraulic_faults.valve import FEATURE_COUNT as VALVE_FEATURE_COUNT
from hydraulic_faults.valve import VALVE_LABELS, ValveModel


def main():
    root = Path(__file__).resolve().parents[1]
    assert not Path(hydraulic_faults.__file__).resolve().is_relative_to(root / "hydraulic_faults")
    assert importlib.util.find_spec("sklearn") is None
    assert importlib.util.find_spec("pytest") is None
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        for kind in ("rolling", "valve"):
            if kind == "rolling":
                def head(labels):
                    return LinearHead(np.array(labels), np.zeros((len(labels), FEATURE_COUNT)), np.zeros(len(labels)))

                model = Classifier(np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT), head([0, 1]),
                                   tuple(head(sorted(labels)) for labels in CONDITIONS.values()))
                channels = [channel.name for channel in CHANNELS]
            else:
                model = ValveModel(np.zeros(VALVE_FEATURE_COUNT), np.ones(VALVE_FEATURE_COUNT), LinearHead(
                    np.array(VALVE_LABELS), np.zeros((4, VALVE_FEATURE_COUNT)), np.zeros(4),
                ))
                channels = ["ps1_bar", "ps2_bar"]
            release, config, pointer = (base / f"{kind}-{name}.json" for name in ("release", "config", "active"))
            atomic_json(release, make_release(kind, model.to_dict(), {}, {"smoke_fixture": True}))
            atomic_json(config, {
                "schema_version": 1, "machines": {"rig": {
                    name: {"range": [-10, 1000], **({"saturation_values": [-10, 1000]} if kind == "valve" else {})}
                    for name in channels}},
                "max_age_seconds": 60, "stale_after_seconds": 5,
                "output_timeout_seconds": 2, "status_interval_seconds": 5,
            })
            activate_release(release, pointer, config)
            now = time.time_ns() // 1_000_000
            if kind == "rolling":
                records = [{"machine_id": "rig", "timestamp_ms": now - (4 - index) * 1000,
                            "sensors": {name: 1.0 for name in channels}} for index in range(5)]
            else:
                stamps = list(range(now - 1490, now + 1, 10))
                records = [{"machine_id": "rig", "timestamp_ms": now,
                            "ps1_bar": [190.0] * 55 + [155.0] * 95, "ps2_bar": [0.0] * 55 + [120.0] * 95,
                            "acquisition": {"clock_id": "fixture", "ps1_timestamp_ms": stamps, "ps2_timestamp_ms": stamps}}]
            result = subprocess.run(
                [sys.executable, "-I", "-m", "hydraulic_faults.cli", "serve", "--active", str(pointer)],
                cwd=base, input=b"\n".join(json.dumps(record).encode() for record in records),
                capture_output=True, timeout=15,
            )
            assert result.returncode == 4, result.stderr
            events = [json.loads(line) for line in result.stdout.splitlines()]
            diagnoses = [event["result"] for event in events if event["event_type"] == "diagnosis"]
            assert diagnoses[-1]["status"] == ("normal" if kind == "rolling" else "diagnosed"), events
    print("Installed runtime-only wheel: rolling and valve service smoke checks passed")


if __name__ == "__main__":
    main()
