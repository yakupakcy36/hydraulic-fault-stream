# Hydraulic Fault Stream

A Python stream processor that detects faults and classifies component conditions from hydraulic sensor measurements. It reports the cooler, valve, pump and accumulator separately, so one sample window can have more than one fault.

## Run the project

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). Download the ZIP from the [UCI dataset page](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) and save it as `data/uci-hydraulic.zip`.

```sh
uv sync --locked --all-extras
uv run --no-sync python -m hydraulic_faults.cli train data/uci-hydraulic.zip --model artifacts/model.json
uv run --no-sync python -m hydraulic_faults.cli replay data/uci-hydraulic.zip --cycle 0 --seconds 8 | uv run --no-sync python -m hydraulic_faults.cli detect --model artifacts/model.json
uv run --no-sync python -m pytest -q
```

Training prints an evaluation report as JSON. Replay emits one recorded sensor sample per second by default. Use `--speed 0` to remove the wait during a functional check. Cycle numbers start at zero.

## Stream contract

`detect` reads one UTF-8 JSON object per line. Use Unix epoch milliseconds for `timestamp_ms`. For live ingestion, an upstream collector should use consistent one-second bucket boundaries and send measurements in increasing timestamp order per machine.

```json
{"machine_id":"rig-01","timestamp_ms":1760000000000,"sensors":{"ps1_bar":160.2,"ps2_bar":110.7,"ps3_bar":2.0,"motor_power_w":2480.0,"flow_l_min":7.2,"ts1_c":35.1,"ts2_c":39.0,"vibration_mm_s":0.42}}
```

UCI replay averages each completed one-second block of the high-rate channels. The model uses the latest five consecutive samples. The first four produce `warming_up`. Later records produce `normal` or `fault` with the four component conditions. An empty `faults` list on a fault record means the binary detector found a fault and the component heads could not localize it. A gap restarts warmup. Duplicate and out-of-order records are discarded. UCI provides sample positions within each cycle, so replay generates its own wall-clock timestamps.

### Stream interruptions

- While the input stream stays open, five seconds without a valid sample emits one `stale` event per machine. The event carries `last_sample_timestamp_ms` and clears the window. The next accepted sample starts a fresh window. The timeout uses local monotonic time.
- Use `detect --stale-after 15` for slow replay such as `replay --speed 0.1`. The timeout can be set from 0.1 to 60 seconds. A machine slot is released after another 60 seconds without a valid sample.
- After a clock reset, the trusted collector can add `"reset": true` to the first sample in the new sequence. The detector clears the window and accepts that sample as the new timestamp reference. Replay uses this marker at the start of each cycle.
- An oversized input line ends the command with exit code 2. Other invalid lines are reported and skipped. A stream with input lines but zero accepted samples also returns 2. A closed output pipe returns 1.

## Evaluation protocol

- Only cycles marked stable in `profile.txt` enter the split. All 60 seconds of a cycle stay on one side. A fault profile is the exact combination of the four component conditions. The split holds out 20% of distinct fault profiles with seed 42 and keeps each profile's cycles together.
- The ten healthy cycles come from one consecutive run. Training uses the first eight and holds out the last two for a same-run check. Measuring false alarms on an independent healthy run would require more data.
- Training fits one standardizer, a class-weighted logistic fault detector, and separate component classifiers. The cooler, valve and pump use class-weighted logistic regression. The accumulator uses a class-weighted decision tree capped at depth 8 with at least 56 windows per leaf. The fixed split reproduces the original reference run. The follow-up model experiments resample stable fault profiles from the same rig.
- The labels describe complete cycles on one test rig. A cycle counts as alerted when at least one window triggers the binary detector. Macro profile recall averages window recall equally across held-out fault profiles. The first-alert statistic is the one-based second of the cycle among alerted fault cycles. Measuring delay from fault onset or false alarms across machines would require different recordings.
- The report separates component predictions made before the binary gate from the final output. For each fault severity, it counts windows already classified as normal by the component head, additional windows labeled normal by the gate, and correct component diagnoses suppressed by the gate.
- The report also checks unstable cycles that are excluded from training and the primary holdout. It separates fault profiles seen in training from unseen profiles. The labels are nominal conditions during settling, so alarms on these cycles measure disagreement with those labels.

## Reference run

Python 3.12.10, NumPy 2.5.4 and scikit-learn 1.9.1. The split used 1,148 training cycles with 114 fault profiles and 301 test cycles with 29 held-out fault profiles. It excluded 756 unstable cycles.

The detector alerted on 14,724 of 16,744 fault windows (87.94%) and on 298 of 299 faulty cycles. It alerted at least once on all 29 held-out fault profiles, with a macro profile window recall of 90.59%. Among alerted fault cycles, the first alarm occurred at cycle second 5 at the median and 11 at the 95th percentile. The two healthy cycles produced zero alerts over 112 windows.

| Component | Macro F1 | Majority baseline |
| --- | ---: | ---: |
| Cooler | 1.0000 | 0.1535 |
| Valve | 0.2715 | 0.0599 |
| Pump | 0.9490 | 0.1548 |
| Accumulator | 0.5632 | 0.0932 |

On the same held-out profiles and with the same binary detector, the previous linear accumulator classifier scored 0.2765 macro F1. Valve severity remains difficult to distinguish with the current one-second aggregates. The training command prints the complete confusion matrices.

Of the 3,864 near-failure accumulator windows, 1,271 were reported as normal. The accumulator tree itself predicted normal on 217. On the other 1,054 windows, the tree predicted the correct severity and the binary gate suppressed it.

| Unstable cycles | Faulty cycles alerted | Nominally healthy cycles alerted |
| --- | ---: | ---: |
| Profiles seen in training | 525 / 532 | 10 / 11 |
| Profiles unseen in training | 120 / 213 | No nominally healthy cycles |

### Model experiments

Four development folds held out complete stable fault profiles with shuffled GroupKFold and seed 17. The last two healthy stable cycles served as the same-run check in every fold. Applying the same window summaries to each second's range and maximum adjacent-sample change from the three 100 Hz pressure channels lifted the linear valve classifier's mean macro F1 from 0.222 to 0.234. Near-failure recall fell in two folds. The stream input contract stays with the original sensor aggregates.

Lowering the binary cutoff from 0.5 to 0.4 raised recall on the reference fault windows from 87.94% to 89.11%. On unstable windows labeled healthy, the alarm fraction rose from 21.59% to 36.53%. Giving the accumulator's near-failure class 1.5 times the weight raised recall for that class on the reference split from 27.82% to 60.12%. It also reduced its precision from 61.53% to 55.44% and lowered accumulator macro F1 from 0.5773 to 0.4858 in one of the four development folds. The reference model retains the original settings.

## Runtime boundaries

- The inference process reads standard input and writes standard output. An edge collector can connect to it over a local pipe.
- Input records are capped at 2,048 bytes. The detector accepts at most 64 machine IDs and keeps five samples per machine. Invalid fields and non-finite values are rejected.
- The UCI reader opens allowlisted ZIP members in place and caps archive, member and row sizes.
- The model is a size-bounded JSON file with validated parameter shapes and bounded tree depth. Inference uses NumPy and can still load earlier version 2 linear models. Training verifies that the exported model reproduces holdout predictions before replacing the output file.
- The current integration has been designed for local replay and monitoring. A PLC, OPC UA server or MQTT broker would require a collector that emits the stream contract above. The classifier emits diagnostics for review.

## Data attribution

Helwig, N., Pignanelli, E., and Schütze, A. (2015), *Condition monitoring of hydraulic systems*. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5CW21) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The input archive SHA-256 used for evaluation is `24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2`. The training command prints the hash it actually uses.
