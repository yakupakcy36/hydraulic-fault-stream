# Strict monitoring service

`serve` is the supervised deployment entry point. `detect`, `valve detect` and `replay` remain useful development interfaces; they do not enforce the complete deployment policy. Field accuracy and installation acceptance remain separate from the software checks below.

## Build and verify a release

From the training checkout with the locked training dependencies installed:

```sh
uv sync --locked --all-extras
uv run --no-sync hydraulic-faults release train data/uci-hydraulic.zip --kind valve --output artifacts/releases/valve-v1.json
uv run --no-sync hydraulic-faults release verify artifacts/releases/valve-v1.json
```

Use `--kind rolling` for the general one-second detector. Training writes into a private staging directory. The final bundle is published only after evaluation, export parity, input stability and source stability checks succeed. An existing release path is never overwritten. Bundle publication uses same-directory rename on Windows and a hard link on POSIX; use a local filesystem supporting those operations. Do not assume these guarantees on an arbitrary network filesystem.

The single JSON bundle contains the model, report, dataset identity, exact train/test cycle indices, Git revision, dirty-working-tree marker, source-file hashes, lock hash and Python/library versions. `release_id` is the SHA-256 of its canonical payload. Startup verifies it before loading the model. Hashes provide integrity against a trusted publisher; they are not digital signatures. Restrict write permissions on releases and the active pointer to the deployment account. Training from a dirty tree is recorded honestly, not presented as an unchanged commit.

## Strict configuration

Each admitted machine must have a rule for every input channel. Rolling rules require a physical `range`, and may specify `flatline_samples` and `saturation_values`. Valve rules require both `range` and `saturation_values` for PS1 and PS2. Saturation values are exact reported ADC rail values, obtained from the collector/sensor specification. They are not normal operating extrema. Any observed configured rail rejects the entire valve capture. This cannot detect undocumented clipping at other values.

The example below is for a synthetic replay collector. Its limits are illustrative, not approved limits for an industrial machine. Replace the machine name and all physical/ADC limits before commissioning.

```json
{
  "schema_version": 1,
  "machines": {
    "uci-hydraulic-rig": {
      "ps1_bar": {"range": [-10, 250], "saturation_values": [-10, 250]},
      "ps2_bar": {"range": [-10, 250], "saturation_values": [-10, 250]}
    }
  },
  "max_age_seconds": 2,
  "stale_after_seconds": 5,
  "output_timeout_seconds": 2,
  "status_interval_seconds": 5
}
```

Valve input requires the synchronized source timestamps described in README. Both modes reject future/old records and unknown IDs before allocating per-machine state. Rolling quality failures clear the inference window. Valve invalid-range/saturation failures do not advance its ordering watermark. A valid but incomplete transition produces `unavailable` and consumes that timestamp. Host and acquisition clocks must be synchronized. A fixed capture is not a clock-synchronization proof.

## Activate and roll back

```sh
hydraulic-faults release activate artifacts/releases/valve-v1.json --config valve-service.json --active artifacts/active.json
hydraulic-faults serve --active artifacts/active.json
```

`serve` reads NDJSON from a trusted collector on stdin and emits versioned NDJSON on stdout. Activation validates the release and configuration, then atomically replaces the active pointer. The pointer snapshots the configuration and binds its hash to the release ID. Changing the original configuration file does not change an activated service.

Rollback uses the same activation command with the previous release and its previous configuration, followed by a supervised restart. A running process deliberately retains its loaded release/configuration until restart. There is no partial hot reload. Invalid activation leaves the previous pointer intact. Concurrent successful activations are last-writer-wins; serialize deployment operations in your supervisor/deployment procedure.

Use a dedicated unprivileged service account. Run the process under an OS supervisor, with restart backoff and an independent monitoring-loss alert. Bound and rotate retained logs outside this process. The repository does not install a system service, configure a PLC or claim a broker integration.

## Event contract

Every normal service event includes `schema_version: 1`, `release_id`, `config_id`, a process-local increasing `sequence` and `emitted_at_ms`. Sequence gaps can occur when repetitive rejection messages are suppressed; these are not durable delivery offsets. Process restarts reset the sequence.

| Event | Meaning |
| --- | --- |
| `started` | Configuration and release loaded; not a healthy machine verdict. |
| `diagnosis` | Contains the existing result under `result`, plus input age when applicable. |
| `rejected` | Bounded reason code and short message; at most one per reason per status interval. |
| `health` | Exact cumulative received/accepted/rejected counts, status counts, rejection reasons, maximum/total processing milliseconds and per-machine time since valid input. |
| `ended` | Input EOF; monitoring has ended. |
| `fatal` | Unrecoverable input/service error. Startup errors go to stderr and may lack release identity. |

`component_evidence` exposes the rolling classifier's component results before the binary gate. `conditions`, `faults` and the operational fault decision retain the gate policy. Raw evidence is not a calibrated probability or an independent safety alarm. A consumer must distinguish component disagreement from a verified fault.

Health events are process telemetry, not collector heartbeats. In the event-driven valve path, silence can mean no switch or a failed collector. The integration must provide a separate collector heartbeat or expected-switch watchdog. `seconds_since_valid_input: null` means the process has received no valid record for that admitted machine. Never retain a diagnosis indefinitely as a current normal-health indication.

## Failure and backpressure contract

| Exit | Meaning for `serve` |
| --- | --- |
| 1 | Output pipe closed. |
| 2 | Invalid startup/configuration, oversized input or other service error. |
| 3 | A stdout/stderr OS write exceeded the configured deadline. |
| 4 | Input EOF, including an empty stream. |

The input reader has a one-record queue and size-bounded reads. Events are written synchronously with an independent deadline watchdog. If output remains blocked, the watchdog terminates the process with code 3 without trying to log to the blocked channel. This deliberately abandons in-flight work; it does not promise delivery of every event. While a write is blocked, diagnostics and health output pause for at most that deadline under normal OS scheduling. The supervisor must detect process loss independently. A partially written final NDJSON line must be discarded by the consumer.

Repeated rejected inputs do not create unbounded logs or queues; exact counts remain in `health`. Processing latency measures admission/inference only, not collector transport or sink latency. Deadline and freshness settings are operational budgets chosen for the installation, not claims of hard-real-time guarantees.

## Verification

CI runs Python 3.12–3.14 on Windows/Linux, lint, boundary type checks, deterministic and generated lifecycle tests, real closed/blocked-pipe subprocess tests, and a wheel smoke check in a fresh NumPy-only environment. The smoke check runs outside the checkout and checks both live model paths without sklearn/pytest installed. Local execution validates the current host; remote CI status must be checked after publishing a branch.

The test suite includes invalid activation preserving the previous version, restoring the prior model/configuration pair, failed training publishing nothing, strict machine admission, rail clipping, timestamp mismatch, model export parity and bounded lifecycle state. Actual sensor limits, OS service installation, collector timing and acceptance of model error rates remain installation responsibilities.
