# Valve robustness evaluation

The archive has been studied repeatedly. All numbers here are development evidence on one hydraulic rig, not a new external test. The clean reference split remains useful for regression but is not an untouched final acceptance set.

## Reproduction

```sh
hydraulic-faults valve compare data/uci-hydraulic.zip --objective clean --report artifacts/clean-comparison.json
hydraulic-faults valve compare data/uci-hydraulic.zip --objective robust --report artifacts/robust-comparison.json
hydraulic-faults valve train data/uci-hydraulic.zip --method lda --model artifacts/valve-baseline.json
hydraulic-faults valve train data/uci-hydraulic.zip --method augmented_lda --model artifacts/valve-robust.json
```

Report paths must be new. Comparison saves all outer train/test cycle indices, inner candidate losses, the seed, support counts, synthetic stress results and paired group-bootstrap intervals. Four outer folds and three inner folds group exact component-condition profiles. All stable healthy cycles stay together as one group. The old training report's healthy-tail split and seed-17 folds are distinct protocols and should not be pooled with this comparison.

Candidates are ordinary LDA, shrinkage LDA, balanced logistic regression with C=1, and LDA trained with fixed synthetic variants. Scalers and estimators fit training indices only. Each augmented training capture contributes the original plus gain factors 0.98 and 1.02 and two independent Gaussian perturbations with standard deviation 0.5 bar (seed 732). Perturbations occur before transition extraction. The model still exports one small linear head and has the same inference dependencies.

The clean objective selects by missed critical captures, then false critical captures, then all errors. The robustness objective applies that lexicographic ordering to the concatenation of clean, +2% gain and 0.5-bar-noise inner validation scenarios. It is an explicit synthetic operating assumption with equal scenario counts, not measured field prevalence. Clipping is excluded from selection because saturated measurements should be rejected by the collector/service. Evaluation noise uses seed 731, separate from training noise.

## Results

The clean objective chose original LDA in all four folds. The robustness objective chose augmented LDA in all four folds. Each stable capture receives one outer-fold prediction.

| Scenario, 1,449 stable captures | Original correct | Augmented correct | Original critical correct / 360 | Augmented critical correct / 360 | Original / augmented false critical |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 1,439 | 1,443 | 359 | 359 | 1 / 1 |
| +2% gain | 682 | 1,442 | 360 | 359 | 205 / 2 |
| Gaussian noise, std 0.5 bar | 1,118 | 1,415 | 352 | 358 | 5 / 0 |
| +2 bar offset | 1,439 | 1,443 | 359 | 359 | 1 / 1 |
| Clipped at 170 bar, before quality rejection | 434 | 703 | 0 | 360 | 0 / 207 |

Augmentation substantially improves the tested gain/noise sensitivity. It does not fix clipping. With the true 170-bar ADC rail configured as a saturation value, strict service rejects that clipped input before advancing its inference state. Unknown clipping levels cannot be inferred reliably from these results.

The paired fault-profile bootstrap's 95% interval for clean accuracy improvement is approximately [-0.069, +0.690] percentage points; it includes zero. Do not claim a statistically established clean-data improvement. The critical recall difference interval is zero because both policies made the same critical correctness decisions in this comparison.

On the original 301-cycle reference split, augmented LDA retains 300 correct captures, 79/80 critical diagnoses and zero false critical diagnoses, including the neighbor-exclusion check. On all 756 unstable nominally normal-valve captures it retains 751 normal predictions and zero critical labels. The older seed-17 development folds retain 359/360 critical diagnoses but introduce one false critical diagnosis where the earlier model had zero. The improved gain behavior also trades 360/360 for 359/360 critical captures while reducing false critical diagnoses from 205 to 2. These tradeoffs remain visible.

Normalizing each pressure response by its final amplitude was also probed. It reduced gain sensitivity but worsened noise and clipping behavior, so it was not integrated. No neural model or online retraining was added.

## Decision

New valve training defaults to `augmented_lda` for its measured gain/noise robustness with unchanged reference-split outcomes. `--method lda` preserves the original training option. Existing model files are not rewritten, and activating a release remains an explicit operation. The comparison command never promotes a model automatically. A claimed improvement beyond these perturbations still needs independent acquisition data.
