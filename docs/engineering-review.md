# Engineering review and offline release plan

Reviewed 2026-11-12 against the working tree, including the uncommitted pilot admission changes. This is a source review and targeted behavioral investigation, not an industrial certification. Real-machine validation is explicitly outside the current executable scope. No production behavior was changed during this review.

Implementation follow-up: strict service admission and deadline handling, release bundles/activation, operational events, installed-wheel CI and generated lifecycle tests are now implemented in [the service workflow](service.md). Nested candidate evaluation and the resulting augmented LDA choice are documented in [the robustness report](robust-valve-evaluation.md). Findings below preserve the original review rationale; legacy `detect` remains a development interface.

## Assessment

The core has useful foundations: bounded records and machine state, independent machine windows, explicit interruption handling, validated JSON model artifacts, atomic individual model replacement, offline/online feature parity, grouped evaluation, Windows/Linux tests and a dependency lock. A compact local process remains an appropriate architecture.

The current project is not yet a completed senior-level release engineering example. Observable failure behavior, artifact lineage, an executable deployment contract, and packaged-system tests need more evidence. Adding a larger model or infrastructure stack would not address those gaps.

## Findings and acceptance criteria

### P1 — Output backpressure stops the monitoring loop

Evidence: `cli.detect` calls `_print_record` synchronously in the same loop as `advance_time`; `_write_output` flushes stdout. A targeted experiment blocked the output callback for 300 ms with a 100 ms stale timeout. Only `warming_up` existed while blocked; `stale` appeared after release. This is an injected blocked-sink experiment, not a measured OS pipe benchmark. Blocking stderr can also stop the loop on rejection logging.

Remedy: define a bounded output queue and deadline, rate-limit repetitive rejection logs, and make blocked-output failure observable to a supervisor through an independent liveness mechanism. Do not silently drop fault events or introduce an unbounded queue. A separate writer alone is insufficient if the main loop can wait indefinitely when its queue fills. The unavailable sink cannot carry its own outage alarm.

Acceptance: a subprocess test with a real unconsumed stdout pipe, input flooding and a bounded runtime must demonstrate bounded memory and the declared exit/watchdog policy. Check recovery and termination on both Windows and Linux. Measure latency under a readable sink separately.

### P1 — Quality configuration is not machine admission control

Evidence: `StreamDetector.ingest` looks up per-machine rules and accepts an unknown machine without rules. A targeted probe with PS1=1 and a registered range [0, 0.5] produced `data_quality` for the registered ID and `warming_up` for an unregistered ID. This is documented behavior, but it is unsafe to assume that passing a quality file enforces a live allowlist.

Remedy: add an explicit strict pilot configuration that declares admitted machine IDs, required quality coverage and freshness/timing requirements. Refuse invalid/incomplete configurations before opening the stream. Keep permissive replay an explicit supported mode. Do not invent universal physical sensor limits.

Acceptance: unknown IDs cannot allocate model state or produce diagnoses in strict mode; a typo cannot bypass quality rules. Missing timing, age limits, malformed configuration and clock skew have deterministic tests. Rollback must restore both configuration and compatible model.

### P1 — Model, evaluation and running predictions are not bound to one release

Evidence: training hashes the dataset, and individual model writes are atomic. The deployed model and stdout diagnostics do not carry a release identity connecting code revision, dependency lock, model hash, feature specification, training configuration, split and report. Evaluation files are ignored under `artifacts/`. In rolling training, model publication occurs before all report computation finishes.

Remedy: produce a local immutable release bundle containing the model, report and manifest. Record SHA-256 values, code revision plus dirty state, lock hash, split indices or their digest, seeds and feature/input/output versions. Validate the complete bundle before promoting an active-release pointer. Verify its hashes at startup and expose the release ID in startup and diagnosis events. A hash establishes identity/integrity against a trusted manifest, not authenticity by itself.

Acceptance: interrupted evaluation leaves the previous active release untouched; mismatched files refuse startup; rollback is exercised; an emitted diagnosis can be traced to its exact bundle. No external registry service is needed at this scale. This adopts the traceability principle of [SLSA provenance](https://slsa.dev/spec/v1.2/provenance/) without claiming SLSA compliance.

### P1 — Reused holdout results are development evidence

Evidence: README documents multiple model, feature and threshold experiments against the same reference split. Grouped folds and neighbor exclusion are helpful, but repeated inspection means the reference holdout should not be presented as an untouched final test. Existing preprocessing is fitted within training subsets; this finding concerns adaptive model selection, not a demonstrated scaler leak.

Remedy: label the reference results as development/reference evidence. Freeze candidate definitions and objectives before the next comparison. Use grouped outer evaluation with candidate selection only inside grouped inner folds. Preserve cycle boundaries; inspect missing classes and skipped folds. Resample whole fault-profile groups for uncertainty estimates, not overlapping one-second windows. Healthy data still contain only one independent run; no bootstrap can manufacture independent healthy machines.

Acceptance: saved split membership, no train/validation group overlap, tuning isolated from outer test, all folds/support/skips visible, and aggregate results include uncertainty and per-severity error tradeoffs. Repartitioning an already studied archive does not erase past analyst exposure. New independent data remain necessary for final external accuracy claims. See [scikit-learn nested evaluation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) and [leakage prevention](https://scikit-learn.org/stable/common_pitfalls.html).

### P2 — Operational health has no structured measurement contract

Evidence: output contains diagnoses and some status events; rejected inputs are text on stderr. There is no versioned operational event schema, periodic counter snapshot, measured processing latency, output saturation signal or model identity. The valve path has no liveness event, appropriately because an idle event source alone cannot reveal a failed collector.

Remedy: define versioned status/error codes, accepted/rejected counts by bounded reason, data age, processing latency and active-machine count. Define collector heartbeat and monitoring-loss semantics separately from machine health. Establish measured budgets on a named test host before claiming an SLA. Start with structured local events and a supervisor; an HTTP metrics server is optional integration work.

Acceptance: loss of input, EOF, process death, invalid-only input and blocked output are distinguishable; operational events cannot be confused with normal machine diagnoses. Labels/IDs cannot create unbounded metrics cardinality. This applies [Google SRE monitoring principles](https://sre.google/sre-book/monitoring-distributed-systems/) to a local stream process.

### P2 — CI validates the source environment, not the full advertised distribution

Evidence: CI runs pytest on Windows/Linux under Python 3.12 and audits dependencies on Linux. `pyproject.toml` permits Python 3.12–3.14. There is no wheel installation smoke test outside the checkout, inference-only environment test, lint/type-check gate, or real blocked-pipe/load test. Existing deterministic state tests are valuable but do not explore long generated action sequences.

Remedy: either test supported Python minors or narrow the advertised range. Build a wheel, install into a clean runtime-only environment, then run both CLI paths outside the source tree with small validated fixtures. Add lint and incremental typing at public boundaries. Use generated state sequences for gaps, resets, duplicates, machine capacity, invalid samples and shutdown. A flat layout is acceptable if installation tests ensure imports actually come from the built package.

Acceptance: installed runtime works without sklearn/pytest; training extras remain separate; malformed artifacts fail cleanly; critical state invariants survive generated sequences. Use [PyPA packaging guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) and [Hypothesis stateful tests](https://hypothesis.readthedocs.io/en/latest/stateful.html). Coverage percentage alone is not the acceptance criterion.

## Better model approaches worth testing

1. Keep pressure-transition features as the primary valve approach when 100 Hz synchronized data exist. Compare current LDA with shrinkage LDA (`lsqr`, `shrinkage="auto"`) and a regularized logistic baseline under the frozen grouped protocol. Correlated response blocks make covariance regularization a plausible candidate, not a proven improvement. All retain compact linear inference. See [scikit-learn LDA shrinkage](https://scikit-learn.org/stable/modules/lda_qda.html).
2. Separate component evidence from alarm policy. The binary gate currently overwrites all component outputs with normal when it predicts no fault. Expose raw component evidence and the gate decision under a versioned output schema first. Evaluate any new alarm policy independently; an OR of all component alarms can increase false positives. Evidence visibility can improve without silently changing the operational alarm threshold.
3. Add controlled amplitude, offset, noise, clipping, missing-channel and timing perturbation experiments. Fit any augmentation only inside training folds. Preserve relative pressure timing; compensating observed edge delay may erase the fault mechanism. Synthetic tests measure robustness to stated assumptions, not new machines.
4. Investigate an explicit abstention decision for unsupported feature shapes only after defining coverage and error objectives. The existing transition check detects incomplete/wrong-direction shapes, not all out-of-distribution captures. Score margin is not calibrated certainty. Fit rejection thresholds within training/inner validation, report retained coverage and missed critical events, and retain a no-rejection baseline. Do not attach an unvalidated confidence percentage.

No candidate is approved by this research. Full nested evaluation does not reverse knowledge already gained from this archive. Field-independent engineering can be improved now; field accuracy remains a separate claim, consistent with [NIST AI RMF measurement principles](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/).

## Implementation order

1. Strict pilot admission/configuration and bounded blocked-output behavior, with regression tests.
2. Versioned operational events and immutable model release bundles, with failure/rollback tests.
3. Packaged-runtime CI, supported-Python matrix and generated state tests.
4. Freeze and automate the model comparison protocol; then test the small linear alternatives and evidence/alarm separation.

Keep the project as a modular local application with pure feature/model functions, a per-machine state layer, and CLI/transport adapters. Avoid Kafka, Kubernetes, microservices, online retraining, a feature store or a Transformer unless a measured requirement justifies them. External monitoring infrastructure and formal industrial safety certification are not implied by passing these offline gates.
