from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic

import numpy as np

from hydraulic_faults.contract import CONDITIONS, SAMPLE_INTERVAL_MS, WINDOW_SIZE, InputError, Sample
from hydraulic_faults.model import Classifier, window_features
from hydraulic_faults.quality import QualityMonitor, SensorRule

MAX_MACHINES = 64
DEFAULT_STALE_SECONDS = 5.0
STALE_RETENTION_SECONDS = 60.0


class MachinePhase(StrEnum):
    WARMING_UP = "warming_up"
    READY = "ready"
    STALE = "stale"


@dataclass
class MachineState:
    timestamp_ms: int
    received_at: float
    phase: MachinePhase = MachinePhase.WARMING_UP
    samples: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    quality: QualityMonitor | None = None


class StreamDetector:
    def __init__(
        self, model: Classifier, stale_after_seconds: float = DEFAULT_STALE_SECONDS,
        *, quality_rules: dict[str, tuple[SensorRule, ...]] | None = None,
    ):
        if not 0.1 <= stale_after_seconds <= 60:
            raise InputError("stale timeout must be between 0.1 and 60 seconds")
        self.model = model
        self.stale_after_seconds = stale_after_seconds
        self.quality_rules = quality_rules or {}
        self.machines: dict[str, MachineState] = {}

    def advance_time(self, now: float) -> list[dict]:
        events = []
        for machine_id, state in list(self.machines.items()):
            idle = now - state.received_at
            if idle >= self.stale_after_seconds and state.phase != MachinePhase.STALE:
                state.samples.clear()
                state.phase = MachinePhase.STALE
                events.append({
                    "machine_id": machine_id,
                    "last_sample_timestamp_ms": state.timestamp_ms,
                    "status": MachinePhase.STALE,
                })
            if idle >= self.stale_after_seconds + STALE_RETENTION_SECONDS:
                del self.machines[machine_id]
        return events

    def ingest(self, sample: Sample, *, now: float | None = None) -> dict:
        now = monotonic() if now is None else now
        state = self.machines.get(sample.machine_id)
        gap = False
        if state is None:
            if len(self.machines) >= MAX_MACHINES:
                raise InputError("machine capacity reached")
            rules = self.quality_rules.get(sample.machine_id)
            quality = QualityMonitor(rules) if rules else None
            state = self.machines[sample.machine_id] = MachineState(sample.timestamp_ms, now, quality=quality)
        else:
            elapsed = sample.timestamp_ms - state.timestamp_ms
            if elapsed <= 0 and not sample.reset:
                raise InputError("duplicate or out-of-order timestamp")
            gap = (
                sample.reset or state.phase == MachinePhase.STALE or elapsed != SAMPLE_INTERVAL_MS
                or now - state.received_at >= self.stale_after_seconds
            )
            if gap:
                state.samples.clear()
                state.phase = MachinePhase.WARMING_UP
                if state.quality is not None:
                    state.quality.reset()
        state.timestamp_ms = sample.timestamp_ms
        state.received_at = now

        result = {"machine_id": sample.machine_id, "timestamp_ms": sample.timestamp_ms}
        if state.quality is not None:
            issues = state.quality.observe(sample.values)
            if issues:
                state.samples.clear()
                state.phase = MachinePhase.WARMING_UP
                return {**result, "status": "data_quality", "issues": issues}
        state.samples.append(sample.values)
        if len(state.samples) < WINDOW_SIZE:
            return {
                **result, "status": state.phase, "samples": len(state.samples), "gap": gap,
            }
        state.phase = MachinePhase.READY
        faulty, codes = self.model.predict(window_features(np.asarray(state.samples)))
        conditions = {
            name: states[int(code)]
            for (name, states), code in zip(CONDITIONS.items(), codes, strict=True)
        }
        faults = [name for name, condition in conditions.items() if condition != "normal"]
        return {
            **result, "status": "fault" if faulty else "normal",
            "conditions": conditions, "faults": faults,
        }
