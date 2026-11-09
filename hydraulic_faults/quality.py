import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from hydraulic_faults.contract import CHANNELS, MACHINE_ID_PATTERN, MAX_SENSOR_MAGNITUDE, InputError, _unique_object

MAX_CONFIG_BYTES = 65_536
CHANNEL_INDEX = {channel.name: index for index, channel in enumerate(CHANNELS)}


@dataclass(frozen=True)
class SensorRule:
    channel_index: int
    value_range: tuple[float, float] | None
    flatline_samples: int | None


def load_quality_rules(path: Path, *, max_machines: int) -> dict[str, tuple[SensorRule, ...]]:
    with path.open("rb") as source:
        raw = source.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise InputError("sensor quality configuration exceeds size limit")
    try:
        config = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise InputError("invalid sensor quality configuration JSON") from exc
    if not isinstance(config, dict) or len(config) > max_machines:
        raise InputError("invalid sensor quality machine configuration")
    result = {}
    for machine, sensors in config.items():
        if not MACHINE_ID_PATTERN.fullmatch(machine) or not isinstance(sensors, dict) or not sensors:
            raise InputError("expected named machines with sensor quality rules")
        rules = []
        for name, options in sensors.items():
            if name not in CHANNEL_INDEX or not isinstance(options, dict) or not options:
                raise InputError("invalid sensor quality rule")
            if options.keys() - {"range", "flatline_samples"}:
                raise InputError("unexpected sensor quality option")
            limits = options.get("range")
            if "range" in options and (
                not isinstance(limits, list) or len(limits) != 2
                or any(type(value) not in (int, float) or not -MAX_SENSOR_MAGNITUDE <= value <= MAX_SENSOR_MAGNITUDE
                       or not math.isfinite(value) for value in limits)
                or limits[0] >= limits[1]
            ):
                raise InputError("sensor quality range requires two finite increasing bounds")
            repeat = options.get("flatline_samples")
            if "flatline_samples" in options and (type(repeat) is not int or not 2 <= repeat <= 3_600):
                raise InputError("flatline_samples must be an integer between 2 and 3600")
            rules.append(SensorRule(CHANNEL_INDEX[name], tuple(limits) if limits is not None else None, repeat))
        result[machine] = tuple(rules)
    return result


@dataclass
class QualityMonitor:
    rules: tuple[SensorRule, ...]
    previous: tuple[float, ...] | None = None
    repeats: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.previous = None
        self.repeats.clear()

    def observe(self, values: tuple[float, ...]) -> dict[str, str]:
        issues = {}
        for rule in self.rules:
            index = rule.channel_index
            value = values[index]
            repeat = rule.flatline_samples
            if repeat is not None:
                previous_count = self.repeats.get(index, 0)
                same = self.previous is not None and value == self.previous[index]
                self.repeats[index] = min(previous_count + 1, repeat) if same else 1
            if rule.value_range is not None and not rule.value_range[0] <= value <= rule.value_range[1]:
                issues[CHANNELS[index].name] = "out_of_range"
            elif repeat is not None and self.repeats[index] == repeat:
                issues[CHANNELS[index].name] = "flatline"
        self.previous = values
        return issues
