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

## Evaluation protocol

- Each training or holdout unit is a whole 60-second cycle. All windows from a cycle stay on the same side of the split.
- The seed is 42. Only cycles marked stable in `profile.txt` enter the split. The 80/20 split is stratified by whether all four components are healthy.
- Training fits one standardizer, a binary fault detector and four component classifiers using class-weighted logistic regression. Each component head is compared with the most frequent class from training.
- Evaluation reflects a single test rig with repeated operating conditions. The labels describe complete cycles. The dataset supports testing classification on completed windows, with limited evidence about fault onset or field false alarms.

## Reference run

Python 3.12.10, NumPy 2.5.4 and scikit-learn 1.9.1. The split used 1,159 training cycles and 290 holdout cycles. It excluded 756 unstable cycles. There were 16,240 holdout windows, of which 112 came from two healthy cycles.

| Fault decision on the development holdout | Fault windows caught | Healthy windows flagged |
| --- | ---: | ---: |
| Initial model, any component predicts a fault | 16,093 / 16,128 (99.78%) | 112 / 112 |
| Separate binary detector | 14,106 / 16,128 (87.46%) | 0 / 112 |

The initial false alarms motivated the binary detector. Both rows use the same development holdout. Two healthy cycles give limited evidence about field false alarms.

| Component | Final macro F1 | Majority baseline |
| --- | ---: | ---: |
| Cooler | 0.9995 | 0.1566 |
| Valve | 0.2684 | 0.0983 |
| Pump | 0.9229 | 0.1735 |
| Accumulator | 0.3469 | 0.0904 |

Valve and accumulator severity remain difficult to distinguish with the current one-second aggregates. The training command prints the complete confusion matrices.

## Runtime boundaries

- The inference process reads standard input and writes standard output. An edge collector can connect to it over a local pipe.
- Input records are capped at 2,048 bytes. The detector accepts at most 64 machine IDs and keeps five samples per machine. Invalid fields and non-finite values are rejected.
- The UCI reader opens allowlisted ZIP members in place and caps archive, member and row sizes.
- The model is a versioned JSON file with fixed shapes and bounded numeric values. Training verifies that the exported model reproduces holdout predictions before replacing the output file.
- The current integration has been designed for local replay and monitoring. A PLC, OPC UA server or MQTT broker would require a collector that emits the stream contract above. The classifier emits diagnostics for review.

## Data attribution

Helwig, N., Pignanelli, E., and Schütze, A. (2015), *Condition monitoring of hydraulic systems*. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5CW21) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The input archive SHA-256 used for evaluation is `24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2`. The training command prints the hash it actually uses.
