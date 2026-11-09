import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from hydraulic_faults.contract import CHANNELS, CONDITIONS, MACHINE_ID_PATTERN, MAX_SENSOR_MAGNITUDE, InputError
from hydraulic_faults.contract import _unique_object
from hydraulic_faults.dataset import CYCLE_SECONDS, MAX_ARCHIVE_BYTES, _profiles, _rows
from hydraulic_faults.model import MAX_MODEL_BYTES, LinearHead, _load_linear_head, _numeric_array

CAPTURE_SAMPLES = 150
SAMPLE_RATE_HZ = 100
BENCH_CAPTURE_START = 890
PRE_EDGE_SAMPLES = 10
POST_EDGE_SAMPLES = 50
BLOCK_SAMPLES = 5
FEATURE_COUNT = 2 * (PRE_EDGE_SAMPLES + POST_EDGE_SAMPLES) // BLOCK_SAMPLES
MAX_CAPTURE_BYTES = 8_192
VALVE_LABELS = sorted(CONDITIONS["valve"])
PRESSURE_FIELDS = ("ps1_bar", "ps2_bar")
FEATURE_SPEC = "ps2_rising_edge_shape_v1"


def transition_features(pressures: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Describe a completed, synchronized PS1/PS2 capture without using its clock position."""
    if pressures.shape != (2, CAPTURE_SAMPLES):
        raise InputError("expected 150 simultaneous samples for both pressure channels")
    outlet = pressures[1]
    above = outlet >= np.median(outlet[:20]) + 20
    sustained = np.flatnonzero(above[:-2] & above[1:-1] & above[2:])
    if not len(sustained):
        return None
    onset = int(sustained[0])
    if onset < 25 or onset + POST_EDGE_SAMPLES > CAPTURE_SAMPLES:
        return None
    reference = np.median(pressures[:, onset - 25:onset - 10], axis=1, keepdims=True)
    response = pressures[:, onset - PRE_EDGE_SAMPLES:onset + POST_EDGE_SAMPLES] - reference
    # The outlet rises while the inlet falls during this type of switch.
    tail = np.median(response[:, -10:], axis=1)
    if tail[0] > -5 or tail[1] < 20:
        return None
    return response.reshape(2, -1, BLOCK_SAMPLES).mean(axis=-1).ravel(), onset


@dataclass(frozen=True)
class ValveModel:
    mean: np.ndarray
    scale: np.ndarray
    head: LinearHead

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.head.predict((features - self.mean) / self.scale)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "feature_spec": FEATURE_SPEC,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "head": self.head.to_dict(),
        }


def load_valve_model(path: Path) -> ValveModel:
    with path.open("rb") as source:
        raw = source.read(MAX_MODEL_BYTES + 1)
    if len(raw) > MAX_MODEL_BYTES:
        raise InputError("valve model exceeds size limit")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["feature_spec"] != FEATURE_SPEC:
            raise InputError("incompatible valve model schema")
        mean = _numeric_array(data["mean"], (FEATURE_COUNT,))
        scale = _numeric_array(data["scale"], (FEATURE_COUNT,))
        if (scale < 1e-12).any():
            raise InputError("invalid valve model scale")
        head = _load_linear_head(data["head"], VALVE_LABELS, feature_count=FEATURE_COUNT)
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as exc:
        raise InputError("invalid valve model artifact") from exc
    return ValveModel(mean, scale, head)


def load_valve_captures(
    path: Path, cycle: int | None = None, *, context_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read bench captures with optional adjacent samples for offline evaluation."""
    if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise InputError("expected a local UCI ZIP within the archive size limit")
    end = BENCH_CAPTURE_START + CAPTURE_SAMPLES
    max_context = min(BENCH_CAPTURE_START, CYCLE_SECONDS * SAMPLE_RATE_HZ - end)
    if type(context_samples) is not int or not 0 <= context_samples <= max_context:
        raise InputError("capture context must stay within the cycle")
    start = BENCH_CAPTURE_START - context_samples
    stop = end + context_samples
    with ZipFile(path) as archive:
        profiles = _profiles(archive)
        if cycle is not None:
            if not 0 <= cycle < len(profiles):
                raise InputError(f"cycle must be between 0 and {len(profiles) - 1}")
            profiles = profiles[cycle:cycle + 1]
        captures = np.empty((len(profiles), 2, stop - start))
        for column, channel in enumerate(CHANNELS[:2]):
            rows = _rows(archive, channel.member, CYCLE_SECONDS * SAMPLE_RATE_HZ)
            try:
                selected = rows if cycle is None else islice(rows, cycle, cycle + 1)
                count = 0
                for count, row in enumerate(selected, 1):
                    if count > len(profiles):
                        raise InputError(f"unexpected cycle count in {channel.member}")
                    values = row[start:stop]
                    if (np.abs(values) > MAX_SENSOR_MAGNITUDE).any():
                        raise InputError(f"invalid pressure magnitude in {channel.member}")
                    captures[count - 1, column] = values
                if count != len(profiles):
                    raise InputError(f"unexpected cycle count in {channel.member}")
            finally:
                rows.close()
    return captures, profiles


def diagnose_capture(model: ValveModel, raw: bytes) -> dict:
    if len(raw) > MAX_CAPTURE_BYTES:
        raise InputError("pressure capture exceeds size limit")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise InputError("invalid pressure capture JSON") from exc
    if not isinstance(data, dict) or data.keys() != {"machine_id", "timestamp_ms", *PRESSURE_FIELDS}:
        raise InputError("expected a machine ID, end timestamp and PS1/PS2 pressure traces")
    machine_id, timestamp = data["machine_id"], data["timestamp_ms"]
    if not isinstance(machine_id, str) or not MACHINE_ID_PATTERN.fullmatch(machine_id):
        raise InputError("invalid machine_id")
    if type(timestamp) is not int or not 0 <= timestamp <= 253_402_300_799_999:
        raise InputError("invalid epoch timestamp")
    pressures = [data[field] for field in PRESSURE_FIELDS]
    if any(
        not isinstance(trace, list) or len(trace) != CAPTURE_SAMPLES
        or any(type(value) not in (int, float) or not -MAX_SENSOR_MAGNITUDE <= value <= MAX_SENSOR_MAGNITUDE
               for value in trace)
        for trace in pressures
    ):
        raise InputError("expected 150 finite bounded pressure samples per channel")
    result = {"machine_id": machine_id, "timestamp_ms": timestamp}
    switch = transition_features(np.asarray(pressures, dtype=float))
    if switch is None:
        return {**result, "status": "unavailable", "reason": "complete_valve_transition_required"}
    features, onset = switch
    return {
        **result, "status": "diagnosed",
        "condition": CONDITIONS["valve"][int(model.predict(features))],
        "switch_offset_ms": onset * (1_000 // SAMPLE_RATE_HZ),
    }
