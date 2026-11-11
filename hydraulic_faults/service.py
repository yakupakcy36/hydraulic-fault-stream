"""Strict local monitoring service with bounded input and output failure deadlines."""

import errno
import json
import math
import os
import stat
import time
from collections import Counter
from io import BufferedReader
from pathlib import Path
from queue import Empty, Queue
from threading import Condition, Thread
from typing import cast

from hydraulic_faults.contract import CHANNELS, MAX_SAMPLE_BYTES, InputError, _unique_object, parse_sample
from hydraulic_faults.quality import parse_quality_rules
from hydraulic_faults.release import digest, encode, load_active, read_json
from hydraulic_faults.stream import StreamDetector
from hydraulic_faults.valve import MAX_CAPTURE_BYTES, ValveDetector


class DeadlineWriter:
    """A stuck OS write kills this CLI process with code 3; never used by library callers implicitly."""

    def __init__(self, timeout: float = 5):
        self.timeout = timeout
        self.condition = Condition()
        self.deadline: float | None = None
        self.stopped = False
        self.thread = Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _watch(self):
        with self.condition:
            while not self.stopped:
                if self.deadline is None:
                    self.condition.wait()
                else:
                    remaining = self.deadline - time.monotonic()
                    if remaining <= 0:
                        # Reporting through the stuck pipe would itself block. Supervisor observes exit 3.
                        os._exit(3)
                    self.condition.wait(timeout=remaining)

    def write(self, record: dict, fd: int = 1):
        data = encode(record) + b"\n"
        with self.condition:
            self.deadline = time.monotonic() + self.timeout
            self.condition.notify()
        try:
            remaining = memoryview(data)
            while remaining:
                try:
                    written = os.write(fd, remaining)
                except OSError as exc:
                    if os.name == "nt" and exc.errno == errno.EINVAL and stat.S_ISFIFO(os.fstat(fd).st_mode):
                        raise BrokenPipeError from exc
                    raise
                if written <= 0:
                    raise BrokenPipeError("output closed")
                remaining = remaining[written:]
        finally:
            with self.condition:
                self.deadline = None
                self.condition.notify()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify()
        self.thread.join(timeout=1)


def validate_service_config(config: dict, kind: str) -> dict:
    fields = {"schema_version", "machines", "max_age_seconds", "stale_after_seconds",
              "output_timeout_seconds", "status_interval_seconds"}
    if (
        not isinstance(config, dict) or config.keys() != fields
        or type(config["schema_version"]) is not int or config["schema_version"] != 1
    ):
        raise InputError("invalid service configuration schema")
    for name, maximum in (("max_age_seconds", 60), ("stale_after_seconds", 60),
                          ("output_timeout_seconds", 10), ("status_interval_seconds", 60)):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0.1 <= value <= maximum:
            raise InputError(f"invalid service setting: {name}")
    rules = parse_quality_rules(config["machines"], max_machines=64)
    expected = {channel.name for channel in CHANNELS} if kind == "rolling" else {"ps1_bar", "ps2_bar"}
    if not rules:
        raise InputError("at least one admitted machine is required")
    for machine, sensors in config["machines"].items():
        if sensors.keys() != expected or any("range" not in options for options in sensors.values()):
            raise InputError(f"all {kind} channels need physical ranges for {machine}")
        if kind == "valve" and any(options.keys() != {"range", "saturation_values"} for options in sensors.values()):
            raise InputError("valve capture quality requires range and saturation_values for each channel")
    return config


def load_service_config(path: Path, kind: str) -> dict:
    return validate_service_config(read_json(path, 65_536), kind)


class ServiceEngine:
    def __init__(self, bundle: dict, model, config: dict):
        self.kind = bundle["payload"]["kind"]
        self.config = validate_service_config(config, self.kind)
        self.release_id = bundle["release_id"]
        self.config_id = digest(encode(config))
        self.counts: Counter = Counter()
        self.reasons: Counter = Counter()
        self.last_seen: dict[str, float] = {}
        self.processing_max_ms = 0.0
        self.processing_total_ms = 0.0
        self.sequence = 0
        if self.kind == "rolling":
            self.detector = StreamDetector(model, config["stale_after_seconds"], quality_rules=parse_quality_rules(
                config["machines"], max_machines=64,
            ))
        else:
            self.detector = ValveDetector(model, require_timing=True, max_age_seconds=config["max_age_seconds"])

    def envelope(self, event_type: str, **data) -> dict:
        self.sequence += 1
        return {"schema_version": 1, "event_type": event_type, "release_id": self.release_id,
                "config_id": self.config_id, "sequence": self.sequence,
                "emitted_at_ms": time.time_ns() // 1_000_000, **data}

    def ingest(self, raw: bytes, *, now: float, now_ms: int) -> dict:
        started = time.perf_counter()
        self.counts["received"] += 1
        reason = "invalid_record"
        try:
            data = json.loads(raw, object_pairs_hook=_unique_object)
            if not isinstance(data, dict) or not isinstance(data.get("machine_id"), str):
                raise InputError("missing machine_id")
            machine = data["machine_id"]
            if machine not in self.config["machines"]:
                reason = "unknown_machine"
                raise InputError("machine is not admitted")
            stamp = data.get("timestamp_ms")
            if type(stamp) is not int:
                raise InputError("invalid timestamp")
            if not 0 <= now_ms - stamp <= self.config["max_age_seconds"] * 1_000:
                reason = "timestamp_age"
                raise InputError("timestamp is stale or in the future")
            if self.kind == "rolling":
                result = self.detector.ingest(parse_sample(raw), now=now)
            else:
                # Validate the full input first, before touching any ordering state.
                from hydraulic_faults.valve import diagnose_capture

                diagnose_capture(self.detector.model, raw, require_timing=True)
                for channel, options in self.config["machines"][machine].items():
                    low, high = options["range"]
                    if any(not low <= value <= high for value in data[channel]):
                        reason = "sensor_range"
                        raise InputError("capture pressure outside configured range")
                    if any(value in options["saturation_values"] for value in data[channel]):
                        reason = "sensor_saturation"
                        raise InputError("capture reached a configured ADC saturation value")
                result = self.detector.ingest(raw, now_ms=now_ms)
            self.last_seen[machine] = now
            self.counts["accepted"] += 1
            self.counts[result["status"]] += 1
            return self.envelope("diagnosis", result=result, data_age_ms=now_ms - stamp)
        except (ValueError, TypeError, RecursionError) as exc:
            self.counts["rejected"] += 1
            self.reasons[reason] += 1
            return self.envelope("rejected", reason=reason, message=str(exc)[:200])
        finally:
            elapsed = (time.perf_counter() - started) * 1_000
            self.processing_total_ms += elapsed
            self.processing_max_ms = max(self.processing_max_ms, elapsed)

    def health(self, now: float) -> dict:
        return self.envelope(
            "health", counts=dict(self.counts), rejection_reasons=dict(self.reasons),
            processing_ms={"total": self.processing_total_ms, "max": self.processing_max_ms},
            seconds_since_valid_input={machine: now - self.last_seen[machine] if machine in self.last_seen else None
                                       for machine in self.config["machines"]},
        )


def run_service(engine: ServiceEngine, source, writer) -> int:
    from hydraulic_faults.cli import _read_records

    limit = MAX_SAMPLE_BYTES if engine.kind == "rolling" else MAX_CAPTURE_BYTES
    inbox: Queue = Queue(maxsize=1)
    Thread(target=_read_records, args=(source, inbox, limit), daemon=True).start()
    writer.write(engine.envelope("started", kind=engine.kind))
    next_health = time.monotonic() + engine.config["status_interval_seconds"]
    reported_reasons: set[str] = set()
    while True:
        now = time.monotonic()
        if now >= next_health:
            writer.write(engine.health(now))
            next_health = now + engine.config["status_interval_seconds"]
            reported_reasons.clear()
        if engine.kind == "rolling":
            for result in engine.detector.advance_time(now):
                writer.write(engine.envelope("diagnosis", result=result))
        try:
            raw = inbox.get(timeout=min(0.1, max(0.001, next_health - time.monotonic())))
        except Empty:
            continue
        if raw is None:
            writer.write(engine.health(time.monotonic()))
            writer.write(engine.envelope("ended", reason="input_eof"))
            # Live monitoring ending is always a loss of monitoring, including clean EOF.
            return 4
        if isinstance(raw, OSError):
            raise raw
        if len(raw) > limit:
            writer.write(engine.envelope("fatal", reason="oversized_record"))
            return 2
        event = engine.ingest(raw, now=time.monotonic(), now_ms=time.time_ns() // 1_000_000)
        if event["event_type"] == "rejected":
            if event["reason"] in reported_reasons:
                continue
            reported_reasons.add(event["reason"])
        writer.write(event)


def serve(active: Path) -> int:
    writer = DeadlineWriter()
    try:
        bundle, model, config = load_active(active)
        engine = ServiceEngine(bundle, model, config)
        writer.timeout = config["output_timeout_seconds"]
        import sys

        return run_service(engine, cast(BufferedReader, sys.stdin.buffer).raw, writer)
    except BrokenPipeError:
        return 1
    except (OSError, ValueError) as exc:
        try:
            writer.write({"schema_version": 1, "event_type": "fatal", "reason": "service_error",
                          "message": str(exc)[:300]}, fd=2)
        except OSError:
            pass
        return 2
    finally:
        writer.close()
