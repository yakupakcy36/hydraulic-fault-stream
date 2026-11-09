import json
import math
import re
from dataclasses import dataclass

SAMPLE_INTERVAL_MS = 1_000
WINDOW_SIZE = 5
MAX_SAMPLE_BYTES = 2_048
MAX_SENSOR_MAGNITUDE = 1_000_000
MACHINE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,32}")


@dataclass(frozen=True)
class Channel:
    member: str
    hz: int
    name: str


CHANNELS = (
    Channel("PS1.txt", 100, "ps1_bar"),
    Channel("PS2.txt", 100, "ps2_bar"),
    Channel("PS3.txt", 100, "ps3_bar"),
    Channel("EPS1.txt", 100, "motor_power_w"),
    Channel("FS1.txt", 10, "flow_l_min"),
    Channel("TS1.txt", 1, "ts1_c"),
    Channel("TS2.txt", 1, "ts2_c"),
    Channel("VS1.txt", 1, "vibration_mm_s"),
)

CONDITIONS = {
    "cooler": {100: "normal", 20: "reduced_efficiency", 3: "near_failure"},
    "valve": {100: "normal", 90: "small_lag", 80: "severe_lag", 73: "near_failure"},
    "pump": {0: "normal", 1: "weak_leakage", 2: "severe_leakage"},
    "accumulator": {130: "normal", 115: "slightly_reduced", 100: "severely_reduced", 90: "near_failure"},
}
NORMAL_CODES = tuple(next(iter(states)) for states in CONDITIONS.values())


class InputError(ValueError):
    """An invalid input record or model."""


@dataclass(frozen=True)
class Sample:
    machine_id: str
    timestamp_ms: int
    values: tuple[float, ...]
    reset: bool = False


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError("duplicate JSON field")
        result[key] = value
    return result


def parse_sample(raw: bytes) -> Sample:
    if len(raw) > MAX_SAMPLE_BYTES:
        raise InputError("sample exceeds size limit")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise InputError("invalid sample JSON") from exc
    required_fields = {"machine_id", "timestamp_ms", "sensors"}
    if not isinstance(data, dict) or data.keys() not in (required_fields, required_fields | {"reset"}):
        raise InputError("expected machine_id, timestamp_ms and sensors, with optional reset")
    machine_id, timestamp = data["machine_id"], data["timestamp_ms"]
    reset = data.get("reset", False)
    if type(reset) is not bool:
        raise InputError("reset must be a boolean")
    if not isinstance(machine_id, str) or not MACHINE_ID_PATTERN.fullmatch(machine_id):
        raise InputError("invalid machine_id")
    if type(timestamp) is not int or not 0 <= timestamp <= 253_402_300_799_999:
        raise InputError("invalid epoch timestamp")
    sensors = data["sensors"]
    if not isinstance(sensors, dict) or sensors.keys() != {channel.name for channel in CHANNELS}:
        raise InputError("unexpected sensor fields")
    values = tuple(sensors[channel.name] for channel in CHANNELS)
    if any(
        type(value) not in (int, float)
        or not -MAX_SENSOR_MAGNITUDE <= value <= MAX_SENSOR_MAGNITUDE
        or not math.isfinite(value)
        for value in values
    ):
        raise InputError("sensor values must be finite bounded numbers")
    return Sample(machine_id, timestamp, tuple(map(float, values)), reset)
