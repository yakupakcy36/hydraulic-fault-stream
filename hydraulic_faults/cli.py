import argparse
import json
import math
import sys
import time
from pathlib import Path
from zipfile import BadZipFile

from hydraulic_faults.contract import CHANNELS, MAX_SAMPLE_BYTES, SAMPLE_INTERVAL_MS, InputError, parse_sample
from hydraulic_faults.dataset import CYCLE_SECONDS, load_cycles
from hydraulic_faults.model import load_model
from hydraulic_faults.stream import StreamDetector


def _print_record(record):
    print(json.dumps(record, allow_nan=False, separators=(",", ":")), flush=True)


def detect(detector: StreamDetector, source) -> int:
    line_number = 0
    while raw := source.readline(MAX_SAMPLE_BYTES + 1):
        line_number += 1
        if len(raw) > MAX_SAMPLE_BYTES:
            print(f"input line {line_number} exceeds size limit", file=sys.stderr)
            return 2
        try:
            result = detector.ingest(parse_sample(raw))
        except InputError as exc:
            print(f"input line {line_number} ignored: {exc}", file=sys.stderr)
        else:
            _print_record(result)
    return 0


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
        _print_record({
            "machine_id": "uci-hydraulic-rig",
            "timestamp_ms": start_ms + index * SAMPLE_INTERVAL_MS,
            "sensors": dict(zip((channel.name for channel in CHANNELS), values.tolist(), strict=True)),
        })


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
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            from hydraulic_faults.train import train_and_evaluate

            print(json.dumps(train_and_evaluate(args.archive, args.model), indent=2, allow_nan=False))
        elif args.command == "replay":
            replay(args.archive, args.cycle, args.seconds, args.speed)
        else:
            return detect(StreamDetector(load_model(args.model)), sys.stdin.buffer)
    except BrokenPipeError:
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
