# Joint learned-array and head-input refit adopted: -2.006% on 50 networks without the old-pair quartic

2026-10-10T07:50Z · decision · adopted · measured

Refitting the learned arrays jointly with the batch-6 head columns (m2, eold, qb), column 6, column 20, column 17 and mean-correction covariance-fold coefficient (kappa_C) from candidate generation 43 (S43)'s point gave -2.0% in PyTorch; local quartic of old pairs (qold) added +0.08% for +0.58% metered FLOPs and was dropped; the protocol was fixed to keep judging ranks out of gradient batches.

Recorded measurement:

    with qold -1.924% (se 0.312, 40 of 50); without qold -2.006% (se 0.252, 45 of 50); 48 clean -2.068%; kappa_C 0.697 at layers 1-14

## Programs

- [jmix9.sh](jmix9.sh)
- [candidate.py](candidate.py)
- [../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/stack/s43_build.sh](../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/stack/s43_build.sh)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../instruments/multiple-learned-input-columns-share-one-standardization-bias-product/code/weights_from_trainer.py](../../../instruments/multiple-learned-input-columns-share-one-standardization-bias-product/code/weights_from_trainer.py)
- [../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a47_kc/kc_from_trainer.py](../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a47_kc/kc_from_trainer.py)
- [../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a52_pld/dly_from_trainer.py](../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/code/opus_reexam_20261001_v1/engineer/a52_pld/dly_from_trainer.py)
- [pairs.py](pairs.py)
- [jmix.py](jmix.py)
- [../../../instruments/native-parameter-export-supports-ten-fitted-dials-gradient-direction/code/opus_reexam_20261001_v1/trainer/export.py](../../../instruments/native-parameter-export-supports-ten-fitted-dials-gradient-direction/code/opus_reexam_20261001_v1/trainer/export.py)
- [export_dials.py](export_dials.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L529](../../TEAM_LOG_2026-10-01_to_10-22.md#L529)
- [TEAM_LOG_2026-10-01_to_10-22.md#L546](../../TEAM_LOG_2026-10-01_to_10-22.md#L546)
- [TEAM_LOG_2026-10-01_to_10-22.md#L557](../../TEAM_LOG_2026-10-01_to_10-22.md#L557)
- [TEAM_LOG_2026-10-01_to_10-22.md#L561](../../TEAM_LOG_2026-10-01_to_10-22.md#L561)
- [TEAM_LOG_2026-10-01_to_10-22.md#L568](../../TEAM_LOG_2026-10-01_to_10-22.md#L568)
- [TEAM_LOG_2026-10-01_to_10-22.md#L570](../../TEAM_LOG_2026-10-01_to_10-22.md#L570)
- [TEAM_LOG_2026-10-01_to_10-22.md#L571](../../TEAM_LOG_2026-10-01_to_10-22.md#L571)
- [TEAM_LOG_2026-10-01_to_10-22.md#L572](../../TEAM_LOG_2026-10-01_to_10-22.md#L572)
- [TEAM_LOG_2026-10-01_to_10-22.md#L573](../../TEAM_LOG_2026-10-01_to_10-22.md#L573)
- [TEAM_LOG_2026-10-01_to_10-22.md#L574](../../TEAM_LOG_2026-10-01_to_10-22.md#L574)
- [TEAM_LOG_2026-10-01_to_10-22.md#L575](../../TEAM_LOG_2026-10-01_to_10-22.md#L575)

<details>
<summary>All result files and records (3)</summary>

- [jmix_C.log](jmix_C.log) · log
- [head_weights_31x128_without_qold.json](../../../instruments/native-builder-exports-learned-head-rows-maps-covariance-coefficient/head_weights_31x128_without_qold.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Joint-refit sample sensitivity: -0.78% native on ten fresh networks versus -2.0% research on 50 ranks](../2026-10-10-joint-refit-sample-sensitivity-0-78-native-ten-fresh/README.md)
- enables: [Refit protocol separates arm sizing and acceptance onto independent fresh sets of 16 networks](../2026-10-10-refit-protocol-separates-arm-sizing-acceptance-onto-independent-fresh/README.md)
- shipped in: [Submission #335147: joint correction refit, layer-15 law diagonal and padded covariance cut](../../../submissions/335147/README.md)
- ← enables: [Joint-refit evaluation: -0.16% native MSE on ten fresh networks versus -0.60% research MSE on 50 ranks](../2026-10-10-joint-refit-evaluation-0-16-native-mse-ten-fresh/README.md)
- ← enables: [Layer-15 law-diagonal head input adopted: -0.418% pooled MSE on 50 networks](../2026-10-10-layer-15-law-diagonal-head-input-adopted-0-418/README.md)
- ← enables: [Mean-correction covariance fold: observed direction reverses much of the existing subtraction](../2026-10-10-mean-correction-covariance-fold-observed-direction-reverses-much-existing/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-joint-learned-array-head-input-refit-adopted-2-006) · [Copy provenance](../../../PROVENANCE.json)

