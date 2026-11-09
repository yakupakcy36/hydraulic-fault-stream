import argparse
import errno
import json
import math
import os
import stat
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from zipfile import BadZipFile

from hydraulic_faults.contract import CHANNELS, MAX_SAMPLE_BYTES, SAMPLE_INTERVAL_MS, InputError, parse_sample
from hydraulic_faults.dataset import CYCLE_SECONDS, load_cycles
from hydraulic_faults.model import load_model
from hydraulic_faults.quality import load_quality_rules
from hydraulic_faults.stream import DEFAULT_STALE_SECONDS, MAX_MACHINES, StreamDetector

_NO_RECORD = object()


def _write_output(text: str) -> None:
    try:
        print(text, flush=True)
    except OSError as exc:
        # Windows can report a closed stdout pipe as EINVAL.
        if os.name == "nt" and exc.errno == errno.EINVAL and stat.S_ISFIFO(os.fstat(sys.stdout.fileno()).st_mode):
            raise BrokenPipeError from exc
        raise


def _print_record(record):
    _write_output(json.dumps(record, allow_nan=False, separators=(",", ":")))


def _read_records(source, inbox: Queue[bytes | OSError | None]) -> None:
    pending = b""
    try:
        while chunk := source.read(MAX_SAMPLE_BYTES + 1):
            pending += chunk
            while b"\n" in pending:
                line, _, pending = pending.partition(b"\n")
                inbox.put(line + b"\n")
                if len(line) + 1 > MAX_SAMPLE_BYTES:
                    return
            if len(pending) > MAX_SAMPLE_BYTES:
                inbox.put(pending)
                return
        if pending:
            inbox.put(pending)
    except OSError as exc:
        inbox.put(exc)
    finally:
        inbox.put(None)


def detect(detector: StreamDetector, source) -> int:
    inbox = Queue(maxsize=1)
    reader = Thread(target=_read_records, args=(source, inbox), daemon=True)
    reader.start()
    poll_seconds = min(0.25, detector.stale_after_seconds)
    line_number = 0
    accepted = False
    while True:
        try:
            raw = inbox.get(timeout=poll_seconds)
        except Empty:
            raw = _NO_RECORD
        now = time.monotonic()
        for event in detector.advance_time(now):
            _print_record(event)
        if raw is _NO_RECORD:
            continue
        if raw is None:
            return 2 if line_number and not accepted else 0
        if isinstance(raw, OSError):
            raise raw
        line_number += 1
        if len(raw) > MAX_SAMPLE_BYTES:
            print(f"input line {line_number} exceeds size limit", file=sys.stderr)
            return 2
        try:
            result = detector.ingest(parse_sample(raw), now=now)
        except InputError as exc:
            print(f"input line {line_number} ignored: {exc}", file=sys.stderr)
        else:
            accepted = True
            _print_record(result)


def replay(archive: Path, cycle: int, seconds: int, speed: float) -> None:
    if not 1 <= seconds <= CYCLE_SECONDS:
        raise InputError(f"seconds must be between 1 and {CYCLE_SECONDS}")
    if not math.isfinite(speed) or (speed != 0 and not 0.1 <= speed <= 100):
        raise InputError("speed must be zero or between 0.1 and 100")
    samples, _ = load_cycles(archive, cycle=cycle)
    start_ms = time.time_ns() // 1_000_000
    monotonic_start = time.monotonic()
    for index, values in enumerate(samples[0, :seconds]):
        if speed:
            target = monotonic_start + index / speed
            time.sleep(max(0.0, target - time.monotonic()))
        record = {
            "machine_id": "uci-hydraulic-rig",
            "timestamp_ms": start_ms + index * SAMPLE_INTERVAL_MS,
            "sensors": dict(zip((channel.name for channel in CHANNELS), values.tolist(), strict=True)),
        }
        if index == 0:
            record["reset"] = True
        _print_record(record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or run a streaming hydraulic fault classifier.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train and evaluate on the UCI hydraulic archive")
    train.add_argument("archive", type=Path)
    train.add_argument("--model", type=Path, required=True)
    playback = commands.add_parser("replay", help="emit one recorded cycle as NDJSON")
    playback.add_argument("archive", type=Path)
    playback.add_argument("--cycle", type=int, required=True, help="zero-based cycle index")
    playback.add_argument("--seconds", type=int, default=CYCLE_SECONDS)
    playback.add_argument("--speed", type=float, default=1, help="clock speed multiplier, zero disables waiting")
    inference = commands.add_parser("detect", help="read NDJSON samples from standard input")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--quality-config", type=Path, help="machine-specific sensor range and flatline rules")
    inference.add_argument(
        "--stale-after", type=float, default=DEFAULT_STALE_SECONDS, help="idle seconds before emitting stale",
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            from hydraulic_faults.train import train_and_evaluate

            _write_output(json.dumps(train_and_evaluate(args.archive, args.model), indent=2, allow_nan=False))
        elif args.command == "replay":
            replay(args.archive, args.cycle, args.seconds, args.speed)
        else:
            # The reader may block waiting for stdin. Avoid a buffered lock during interpreter shutdown.
            rules = load_quality_rules(args.quality_config, max_machines=MAX_MACHINES) if args.quality_config else None
            detector = StreamDetector(load_model(args.model), args.stale_after, quality_rules=rules)
            return detect(detector, sys.stdin.buffer.raw)
    except BrokenPipeError:
        # Prevent Python's shutdown flush from writing to the closed pipe again.
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())
        return 1
    except ModuleNotFoundError as exc:
        if exc.name != "sklearn":
            raise
        print("training requires the train extra", file=sys.stderr)
        return 2
    except (ValueError, OSError, BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
