from collections import deque
from dataclasses import dataclass, field

import numpy as np

from hydraulic_faults.contract import CONDITIONS, SAMPLE_INTERVAL_MS, WINDOW_SIZE, InputError, Sample
from hydraulic_faults.model import Classifier, window_features

MAX_MACHINES = 64


@dataclass
class MachineState:
    timestamp_ms: int | None = None
    samples: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))


class StreamDetector:
    def __init__(self, model: Classifier):
        self.model = model
        self.machines: dict[str, MachineState] = {}

    def ingest(self, sample: Sample) -> dict:
        state = self.machines.get(sample.machine_id)
        if state is None:
            if len(self.machines) >= MAX_MACHINES:
                raise InputError("machine capacity reached")
            state = self.machines[sample.machine_id] = MachineState()

        gap = False
        if state.timestamp_ms is not None:
            elapsed = sample.timestamp_ms - state.timestamp_ms
            if elapsed <= 0:
                raise InputError("duplicate or out-of-order timestamp")
            if elapsed != SAMPLE_INTERVAL_MS:
                state.samples.clear()
                gap = True
        state.timestamp_ms = sample.timestamp_ms
        state.samples.append(sample.values)

        result = {"machine_id": sample.machine_id, "timestamp_ms": sample.timestamp_ms}
        if len(state.samples) < WINDOW_SIZE:
            return {
                **result, "status": "warming_up", "samples": len(state.samples), "gap": gap,
            }
        codes = self.model.predict(window_features(np.asarray(state.samples)))
        conditions = {
            name: states[int(code)]
            for (name, states), code in zip(CONDITIONS.items(), codes, strict=True)
        }
        faults = [name for name, condition in conditions.items() if condition != "normal"]
        return {
            **result, "status": "fault" if faults else "normal",
            "conditions": conditions, "faults": faults,
        }
