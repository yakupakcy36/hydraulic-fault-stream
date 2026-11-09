# Hydraulic Fault Stream

A Python stream processor that detects faults and classifies component conditions from hydraulic sensor measurements. It reports the cooler, valve, pump and accumulator separately, so one sample window can have more than one fault.

## Run the project

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). Download the ZIP from the [UCI dataset page](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) and save it as `data/uci-hydraulic.zip`.

```sh
uv sync --locked --all-extras
uv run --no-sync python -m hydraulic_faults.cli train data/uci-hydraulic.zip --model artifacts/model-temporal.json
uv run --no-sync python -m hydraulic_faults.cli replay data/uci-hydraulic.zip --cycle 0 --seconds 8 | uv run --no-sync python -m hydraulic_faults.cli detect --model artifacts/model-temporal.json
uv run --no-sync python -m pytest -q
```

Training prints an evaluation report as JSON. Replay emits one recorded sensor sample per second by default. Use `--speed 0` to remove the wait during a functional check. Cycle numbers start at zero.

## Stream contract

`detect` reads one UTF-8 JSON object per line. Use Unix epoch milliseconds for `timestamp_ms`. For live ingestion, an upstream collector should use consistent one-second bucket boundaries and send measurements in increasing timestamp order per machine.

```json
{"machine_id":"rig-01","timestamp_ms":1760000000000,"sensors":{"ps1_bar":160.2,"ps2_bar":110.7,"ps3_bar":2.0,"motor_power_w":2480.0,"flow_l_min":7.2,"ts1_c":35.1,"ts2_c":39.0,"vibration_mm_s":0.42}}
```

UCI replay averages each completed one-second block of the high-rate channels. The model uses the latest five consecutive samples. The first four produce `warming_up`. Later records produce `normal` or `fault` with the four component conditions. An empty `faults` list on a fault record means the binary detector found a fault and the component heads could not localize it. A gap restarts warmup. Duplicate and out-of-order records are discarded. UCI provides sample positions within each cycle, so replay generates its own wall-clock timestamps.

The valve head combines its current scores with recent scores from the same machine. It uses the current window alone after warmup and preserves an immediate near-failure valve diagnosis when the binary detector reports a fault. A gap, explicit reset, stale stream or rejected quality sample clears the valve score history.

### Stream interruptions

- While the input stream stays open, five seconds without a valid sample emits one `stale` event per machine. The event carries `last_sample_timestamp_ms` and clears the window. The next accepted sample starts a fresh window. The timeout uses local monotonic time.
- Use `detect --stale-after 15` for slow replay such as `replay --speed 0.1`. The timeout can be set from 0.1 to 60 seconds. A machine slot is released after another 60 seconds without a valid sample.
- After a clock reset, the trusted collector can add `"reset": true` to the first sample in the new sequence. The detector clears the window and accepts that sample as the new timestamp reference. Replay uses this marker at the start of each cycle.
- An oversized input line ends the command with exit code 2. Other invalid lines are reported and skipped. A stream with input lines but zero accepted samples also returns 2. A closed output pipe returns 1.

### Sensor quality checks

Pass `detect --quality-config quality.json` to enable rules for specific machine IDs and sensor names. This example checks PS1 in the UCI replay for five identical consecutive one-second values.

```json
{"uci-hydraulic-rig":{"ps1_bar":{"flatline_samples":5}}}
```

A sensor rule can also specify an inclusive `"range": [minimum, maximum]` in the sensor's units. Set ranges from the machine's operating limits and use flatline rules on channels expected to vary. The example's five-sample PS1 rule flagged zero original windows in the UCI archive. Several other channels contained naturally constant windows.

Each rejected reading produces `status: "data_quality"` and an `issues` map from sensor names to `flatline` or `out_of_range`. The detector clears its inference window and resumes after five consecutive usable samples. Gaps and resets also clear the flatline history. The same NDJSON input and model files work with or without this configuration. In a replay of healthy cycle 1795, freezing PS1 produced seven fault decisions without quality rules. With the example rule, all 56 full-window positions in the frozen replay reported a quality issue. The unmodified replay produced 56 normal decisions in both modes.

## Valve switch diagnosis

Use the event model when a collector can supply synchronized 100 Hz inlet and outlet pressure. It returns one valve severity per completed switch capture. The rolling detector still serves streams of one-second aggregates.

```sh
uv run --no-sync python -m hydraulic_faults.cli valve train data/uci-hydraulic.zip --model artifacts/valve-transition-model.json
uv run --no-sync python -m hydraulic_faults.cli valve replay data/uci-hydraulic.zip --cycle 211 | uv run --no-sync python -m hydraulic_faults.cli valve detect --model artifacts/valve-transition-model.json
```

`valve detect` reads one UTF-8 NDJSON object per capture with `machine_id`, `timestamp_ms` for the capture end, and `ps1_bar` and `ps2_bar` arrays of exactly 150 synchronized samples in bars. Start the capture early enough to retain at least 25 samples before the outlet rises and at least 50 samples from the start of the rise. An incomplete transition or a capture without the paired inlet drop returns `unavailable`. A complete capture returns the diagnosis and the switch offset in milliseconds from the start of the capture.

Use a shared acquisition clock for PS1 and PS2. At the collector, pair samples using the hardware sample counter or source timestamps from that clock and verify 150 consecutive pairs at 10 ms intervals. Discard a capture if either channel has missing, duplicate or mismatched indices. The single capture-end timestamp in NDJSON carries no per-channel timing information, so the detector relies on the collector for this check. Preserve the measured delay between inlet and outlet responses because it is part of the valve signal.

UCI replay takes the capture from 8.90 to 10.40 seconds of the fixed cycle. Feature extraction aligns both traces to the observed outlet rise and compares the pressure response against its local baseline. The linear discriminant model receives pressure shape features. Capture timestamps and the cycle clock are used for reporting. The inlet drop and outlet rise are described in the [published valve experiment](https://www.mdpi.com/2227-9717/7/9/589).

| Valve method on the same 301 stable test cycles | Correct critical cycles | Noncritical cycles labeled critical |
| --- | ---: | ---: |
| Rolling detector by cycle second 10 | 34 / 80 | 170 / 221 |
| Rolling detector over the full cycle | 80 / 80 | 202 / 221 |
| Switch capture available at cycle second 10.4 | 79 / 80 | 0 / 221 |

The event model classified 300 of 301 captures correctly with macro F1 of 0.9973. Four grouped development folds produced 359 correct critical diagnoses out of 360, with zero false critical diagnoses. Removing nearby training records gave the same holdout results. Shifting the capture by 150 ms in either direction produced identical features on all 2,205 recordings. All 756 unstable records have nominally normal valve labels. The model reported normal in 751 of those captures and gave zero critical labels. These results cover one test rig and its fixed load schedule.

The training report also tests relative channel skew. It shifts the PS1 source slice by one or two real samples in either direction while keeping PS2 fixed. Positive offsets use later PS1 samples. The check measures model sensitivity to deliberately mispaired data on the stable holdout.

| PS1 source offset | Correct captures | Correct critical captures | False critical diagnoses |
| --- | ---: | ---: | ---: |
| −20 ms | 193 / 301 | 72 / 80 | 0 / 221 |
| −10 ms | 280 / 301 | 78 / 80 | 0 / 221 |
| 0 ms | 300 / 301 | 79 / 80 | 0 / 221 |
| +10 ms | 276 / 301 | 79 / 80 | 0 / 221 |
| +20 ms | 196 / 301 | 79 / 80 | 7 / 221 |

All shifted captures passed the pressure shape check. These offsets were injected into recorded data. Field validation requires measuring channel timing at acquisition.

## Evaluation protocol

- Only cycles marked stable in `profile.txt` enter the split. All 60 seconds of a cycle stay on one side. A fault profile is the exact combination of the four component conditions. The split holds out 20% of distinct fault profiles with seed 42 and keeps each profile's cycles together.
- The ten healthy cycles come from one consecutive run. Training uses the first eight and holds out the last two for a same-run check. Measuring false alarms on an independent healthy run would require more data.
- Training fits one standardizer, a class-weighted logistic fault detector, and separate component classifiers. The cooler, valve and pump use class-weighted logistic regression. The accumulator uses a class-weighted decision tree capped at depth 8 with at least 56 windows per leaf. The fixed split reproduces the original reference run. The follow-up model experiments resample stable fault profiles from the same rig.
- The labels describe complete cycles on one test rig. A cycle counts as alerted when at least one window triggers the binary detector. Macro profile recall averages window recall equally across held-out fault profiles. The first-alert statistic is the one-based second of the cycle among alerted fault cycles. Measuring delay from fault onset or false alarms across machines would require different recordings.
- The report separates component predictions made before the binary gate from the final output. For each fault severity, it counts windows already classified as normal by the component head, additional windows labeled normal by the gate, and correct component diagnoses suppressed by the gate.
- The report also checks unstable cycles that are excluded from training and the primary holdout. It separates fault profiles seen in training from unseen profiles. The labels are nominal conditions during settling, so alarms on these cycles measure disagreement with those labels.
- Each fault profile reports cycle support, missed cycles and window recall. Critical diagnosis reports the first correct severity among diagnosed cycles and counts cycles without one. A healthy alarm episode is a run of fault decisions within a cycle. The per-hour rate uses only seconds with a full inference window. These short, reset-separated recordings give an alarm load for the replay.
- The archive guard refits on the same training split after removing faulty training cycles within ten archive positions of any holdout cycle. It evaluates on the original holdout and preserves the same-run healthy split. The result measures sensitivity to removing nearby recordings and reducing the training set.

## Reference run

Python 3.12.10, NumPy 2.5.4 and scikit-learn 1.9.1. The split used 1,148 training cycles with 114 fault profiles and 301 test cycles with 29 held-out fault profiles. It excluded 756 unstable cycles.

The detector alerted on 14,724 of 16,744 fault windows (87.94%) and on 298 of 299 faulty cycles. It alerted at least once on all 29 held-out fault profiles, with a macro profile window recall of 90.59%. Among alerted fault cycles, the first alarm occurred at cycle second 5 at the median and 11 at the 95th percentile. The two healthy cycles produced zero alerts over 112 windows.

| Component | Macro F1 | Majority baseline |
| --- | ---: | ---: |
| Cooler | 1.0000 | 0.1535 |
| Valve | 0.2861 | 0.0599 |
| Pump | 0.9490 | 0.1548 |
| Accumulator | 0.5632 | 0.0932 |

On the same held-out profiles and with the same binary detector, the previous linear accumulator classifier scored 0.2765 macro F1. The valve head scored 0.2715 macro F1 before combining scores across windows. It now correctly classifies 1,761 of 4,480 near-failure windows (39.31%), up from 1,491 (33.28%). It still reports the correct severity at least once in all 80 near-failure cycles, with a median first correct diagnosis at cycle second 11 and a p95 at second 12. Correct small-lag classifications fell from 395 to 352 of 3,920 windows. The training command prints the complete confusion matrices.

Of the 3,864 near-failure accumulator windows, 1,271 were reported as normal. The accumulator tree itself predicted normal on 217. On the other 1,054 windows, the tree predicted the correct severity and the binary gate suppressed it.

| Unstable cycles | Faulty cycles alerted | Nominally healthy cycles alerted |
| --- | ---: | ---: |
| Profiles seen in training | 525 / 532 | 10 / 11 |
| Profiles unseen in training | 120 / 213 | No nominally healthy cycles |

The near-failure accumulator profile with all other components nominal accounts for 211 of the 213 unseen unstable cycles. It produced alarms in 118 cycles and 9.76% of windows, with the correct critical diagnosis in 101 cycles. The other two profiles contain one cycle each and alerted in every window. On the stable holdout, the system correctly diagnosed near-failure accumulator severity at least once in 63 of 69 cycles. The median first correct diagnosis was cycle second 10 among those 63 cycles.

The archive guard removed 437 of 1,148 training cycles. On the same stable holdout, fault alarms still covered 298 of 299 cycles, and accumulator macro F1 fell from 0.5632 to 0.4626.

### Model experiments

Four development folds held out complete stable fault profiles with shuffled GroupKFold and seed 17. The last two healthy stable cycles served as the same-run check in every fold. Applying the same window summaries to each second's range and maximum adjacent-sample change from the three 100 Hz pressure channels lifted the linear valve classifier's mean macro F1 from 0.222 to 0.234. Near-failure recall fell in two folds. The stream input contract stays with the original sensor aggregates.

In the same folds, the valve timing experiment measured the dominant pressure edge within each completed second, its local 10–90% transition time and differences in edge time across PS1–PS3. The timing features lifted mean macro F1 from 0.222 to 0.226 with the same fault detector. Combining timing with amplitude summaries scored 0.239, compared with 0.234 for amplitude summaries alone. In the first fold, near-failure recall fell from 37.13% to 34.89% with timing and to 33.59% with the combined features. The reference model uses the original features.

Lowering the binary cutoff from 0.5 to 0.4 raised recall on the reference fault windows from 87.94% to 89.11%. On unstable windows labeled healthy, the alarm fraction rose from 21.59% to 36.53%. Giving the accumulator's near-failure class 1.5 times the weight raised recall for that class on the reference split from 27.82% to 60.12%. It also reduced its precision from 61.53% to 55.44% and lowered accumulator macro F1 from 0.5773 to 0.4858 in one of the four development folds. The reference model retains the original settings.

In a follow-up experiment, averaging valve scores across consecutive windows improved window recall but sometimes removed the only correct diagnosis in a cycle. Preserving every immediate near-failure valve result avoids that failure. With 45% weight on the current score and 55% on recent evidence, near-failure window recall improved in all four development folds and averaged 33.00%, up from 26.92%. Critical cycle coverage stayed the same across all four folds. In the first fold, precision for this class dipped from 31.13% to 30.84%. The paired reference run uses the same binary detector. That detector's quadratic and boosted alternatives raised alarms on 93.83% and 100% of nominally healthy unstable windows, compared with 21.59% for the original detector.

## Runtime boundaries

- The inference process reads standard input and writes standard output. An edge collector can connect to it over a local pipe.
- Input records are capped at 2,048 bytes. The detector accepts at most 64 machine IDs and keeps five samples per machine. Invalid fields and non-finite values are rejected.
- The UCI reader opens allowlisted ZIP members in place and caps archive, member and row sizes.
- The model is a size-bounded JSON file with validated parameter shapes and bounded tree depth. Version 4 stores the valve evidence weight. Inference uses NumPy and accepts version 2 linear models and version 3 tree models. Training verifies the exported model against ordered holdout windows before replacing the output file.
- The current integration has been designed for local replay and monitoring. A PLC, OPC UA server or MQTT broker would require a collector that emits the stream contract above. The classifier emits diagnostics for review.

## Data attribution

Helwig, N., Pignanelli, E., and Schütze, A. (2015), *Condition monitoring of hydraulic systems*. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5CW21) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The input archive SHA-256 used for evaluation is `24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2`. The training command prints the hash it actually uses.
