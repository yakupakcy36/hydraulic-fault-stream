# Supervised pilot operation

The implemented strict deployment entry point is now [`serve` with an activated release](service.md). Use that workflow for bounded output failure, required machine admission, acquisition checks, operational events and rollback. Commands below describe legacy development interfaces and integration responsibilities.

## Scope and remaining release blockers

This release is a local diagnostic process. There is no deployed collector, broker, service supervisor or field validation in this repository. A passing software test suite cannot establish accuracy on an industrial installation.

The rolling valve head has substantial false critical diagnoses. The accumulator head and binary gate miss critical cases, especially unstable unseen profiles. Prefer the separate pressure-transition model for a valve pilot when synchronized 100 Hz measurements are available. Its 79/80 critical diagnoses and zero false critical diagnoses on 221 noncritical holdout cycles come from one rig. They are not a field reliability guarantee. Keep independent maintenance and safety controls active.

## Collector contract

- Assign a stable machine ID and use a shared acquisition clock. Keep the detector host clock synchronized with the collector. The age checks reject any future timestamp; choose a local clock arrangement that satisfies this contract.
- For the rolling path, emit completed one-second aggregates with consistent bucket boundaries. Configure physical sensor ranges for each admitted machine using `--quality-config`. Unknown IDs do not inherit another machine's rules; restrict the collector to configured IDs.
- For the valve path, send 150 original paired pressure readings and their source timestamps. Supply `acquisition.clock_id`, `acquisition.ps1_timestamp_ms` and `acquisition.ps2_timestamp_ms`. Do not reconstruct timestamps from network arrival time or align pressure edges independently. Verify the hardware's relative channel skew before connecting it.
- The capture end means the last sample's source timestamp. For end time `T`, each timestamp array is `[T-1490, T-1480, ..., T]`. Clock IDs identify a shared source; the software cannot verify that hardware honored that claim.
- If acquisition drops a sample, discard the affected capture. Never interpolate a missing value to satisfy this contract. The 1500 ms observation window includes the detected transition and its response.

Example commands with a collector writing NDJSON to standard input:

```sh
hydraulic-faults detect --model artifacts/model-temporal.json --quality-config quality.json --max-age-seconds 2 --stale-after 5
hydraulic-faults valve detect --model artifacts/valve-transition-model.json --require-timing --max-age-seconds 2
```

The two-second age limit is an example admission budget, not a measured installation SLA. Measure transport and processing latency before choosing it. Validation flags are optional to preserve historical replay; include them in the supervised service command.

## Process operation and failures

Run under a dedicated unprivileged account with a read-only model file and locked dependencies (`uv sync --locked --all-extras`). Use a supervisor to capture stderr, detect exit/EOF, and restart with backoff. Limit access to the input pipe to the trusted collector. The program opens no network listener and requires no paid API.

Stdout contains diagnostics; stderr contains rejected-record errors. Monitor both. Invalid records are skipped; an all-invalid input exits 2. Oversized records stop ingestion with exit 2. Closed output exits 1. Exit 0 on EOF means the stream ended successfully, not that the machine is healthy. Treat EOF or process death as lost monitoring.

The rolling detector emits `stale` on local idle timeout while input remains open. Valve captures are event-driven and have no idle heartbeat: an independent collector heartbeat and expected-switch watchdog are required to distinguish no switch from a dead collector. Never retain the last diagnosis indefinitely as a current health state. `unavailable`, invalid data and warming up are not normal diagnoses.

Valve ordering state is bounded to 64 machine IDs and persists until restart. Restart clears ordering history; age checking still rejects an old backlog. Plan a process restart when changing the admitted machine set. Do not use machine-ID churn to bypass this limit. The collector owns clock-reset recovery.

Record the deployed commit, dependency lock, SHA-256 of the approved model, command flags and collector configuration. Approve model artifacts before copying them into the service account's read-only directory. Retain the preceding artifact/configuration pair for rollback. Check representative replay and rejection cases before resuming the pilot after a change.

## Evidence required before broader production release

Collect independent healthy operation across machines, loads and temperatures, and labeled fault episodes with known onset times. Define acceptable false alarms per operating hour, per-severity missed-event rate and delay before reviewing this new evaluation set. Report denominators and uncertainty as well as averages. Do not tune on the final acceptance set.

Verify real clock skew, sensor dropouts, repeated packets, collector restarts, slow or blocked output, process death and rollback on the target host. Verify that the operator sees loss of monitoring. Move beyond a supervised pilot only after the installation owner accepts those measured results and the operating procedure. No such field acceptance or external deployment has been performed by this change.
