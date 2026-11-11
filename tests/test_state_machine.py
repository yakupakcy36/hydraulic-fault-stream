import json

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from test_stream import _model, _record

from hydraulic_faults.contract import WINDOW_SIZE, InputError, parse_sample
from hydraulic_faults.stream import MAX_MACHINES, MachinePhase, StreamDetector


class StreamLifecycle(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.detector = StreamDetector(_model(), stale_after_seconds=2)
        self.clock = 0.0

    @rule(machine=st.integers(0, 70), delta=st.sampled_from([-1000, 0, 1000, 3000]), reset=st.booleans())
    def ingest(self, machine, delta, reset):
        name = f"rig-{machine}"
        state = self.detector.machines.get(name)
        stamp = max(0, (state.timestamp_ms if state else 100_000) + delta)
        record = {**_record(stamp, name), "reset": reset}
        sample = parse_sample(json.dumps(record).encode())
        if state is not None and stamp <= state.timestamp_ms and not reset:
            previous = (state.timestamp_ms, state.received_at, len(state.samples))
            with pytest.raises(InputError):
                self.detector.ingest(sample, now=self.clock)
            assert previous == (state.timestamp_ms, state.received_at, len(state.samples))
        elif state is None and len(self.detector.machines) == MAX_MACHINES:
            with pytest.raises(InputError, match="capacity"):
                self.detector.ingest(sample, now=self.clock)
        else:
            previous_length = len(state.samples) if state else 0
            should_restart = state is None or reset or delta != 1000 or self.clock - state.received_at >= 2
            result = self.detector.ingest(sample, now=self.clock)
            expected = 1 if should_restart else min(WINDOW_SIZE, previous_length + 1)
            assert len(self.detector.machines[name].samples) == expected
            assert (result["status"] == "warming_up") == (expected < WINDOW_SIZE)

    @rule(elapsed=st.sampled_from([0.1, 1.0, 2.0, 63.0]))
    def advance(self, elapsed):
        self.clock += elapsed
        events = self.detector.advance_time(self.clock)
        assert len({event["machine_id"] for event in events}) == len(events)
        assert not self.detector.advance_time(self.clock)

    @invariant()
    def bounded_and_consistent(self):
        assert len(self.detector.machines) <= MAX_MACHINES
        for state in self.detector.machines.values():
            assert len(state.samples) <= WINDOW_SIZE
            if state.phase == MachinePhase.STALE:
                assert not state.samples and state.valve_scores is None
            if state.phase == MachinePhase.READY:
                assert len(state.samples) == WINDOW_SIZE


TestStreamLifecycle = StreamLifecycle.TestCase
TestStreamLifecycle.settings = settings(max_examples=50, stateful_step_count=100, deadline=None, derandomize=True)
