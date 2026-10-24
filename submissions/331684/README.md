# Submission #331684: packed frozen covariance-edge correction at layers 8-14 with full lower-moment support

2026-09-20T20:10:53Z · submission · adopted · measured

An independent frozen covariance-edge model (a fixed amplitude-one pair-local covariance correction at layers 8 to 14 on the full lower-moment trajectory of #331672, with the original final readout and no newly fitted readout coefficients) passed its native development, 14-net panel and 8-net confirmation stages; because #331676 had just failed on residual time, only its lossless static-packed derivative was uploaded. Accepted 2026-09-20T20:10:53.544Z, it completed all 100 networks and was graded adjusted 3.5844144212297072e-9, raw 1.949715441540434e-8, the new bank, 0.453755% better than #331682; the old strict covariance and response proxy gates of this model remain recorded as failed.

Recorded measurement:

    official adjusted 3.5844144212297072e-9, raw 1.949715441540434e-8; 100/100 completed, 0 failed; +0.453755% adjusted over #331682, 28/50 public adjusted wins and 32/50 raw wins; local development36/38 pooled adjusted ratio 0.982783220745; panel14 ratio 0.982838708008 (12/14 wins, worst 1.023493156747); confirmation8 ratio 0.982181085851 (6/8, worst 1.025721997737; mean 3.62422917166e-9 vs 3.68998061953e-9); mean cold residual ratio 0.683406108864, max packed residual 0.219410453 s, pooled total-wall ratio 1.018301950425; archive 352348 bytes sha256 9fa02622fe0e1fb387594be0ad964c55117d12498c0e5a013557b4a851e1d149; source sha256 facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0 (uncompressed model 04dd7586ac6826f9aef65d873b719d05184355dd2dc6552f32d6046c066d0180)

Public score: 3.5844144212297072e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [../../experiments/learned-correction/direct-native-16-coefficient-covariance-head-optimization-fails-training/code/full_lower_covariance_edge_native_v1/candidate.py](../../experiments/learned-correction/direct-native-16-coefficient-covariance-head-optimization-fails-training/code/full_lower_covariance_edge_native_v1/candidate.py)

## Evidence

- [OFFICIAL_RESULT.md#L3](OFFICIAL_RESULT.md#L3)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L7](RESULT.md#L7)
- [RESULT.md#L14](RESULT.md#L14)
- [RESULT.md#L15](RESULT.md#L15)
- [RESULT.md#L22](RESULT.md#L22)
- [RESULT.md#L3](../../experiments/covariance-structure/frozen-covariance-edge-correction-passes-panel14-ratio-0-982838708008/RESULT.md#L3)
- [RESULT.md#L18](../../experiments/covariance-structure/frozen-covariance-edge-correction-passes-panel14-ratio-0-982838708008/RESULT.md#L18)
- [ACCURACY_RESULT.md#L4](../../experiments/covariance-structure/frozen-covariance-edge-correction-passes-panel14-ratio-0-982838708008/ACCURACY_RESULT.md#L4)
- [REEXAMINATION_INDEX_20260919.md#L846](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L846)
- [REEXAMINATION_INDEX_20260919.md#L855](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L855)
- [REEXAMINATION_INDEX_20260919.md#L867](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L867)

<details>
<summary>All result files and records (14)</summary>

- [OFFICIAL_RESULT.md](OFFICIAL_RESULT.md) · record
- [RESULT.md](RESULT.md) · record
- [release_interpretation.json](release_interpretation.json) · record
- [ACCURACY_RESULT.md](../../experiments/covariance-structure/frozen-covariance-edge-correction-passes-panel14-ratio-0-982838708008/ACCURACY_RESULT.md) · result
- [RESULT.md](../../experiments/covariance-structure/frozen-covariance-edge-correction-passes-panel14-ratio-0-982838708008/RESULT.md) · result
- [official_best_331684.json](official_best_331684.json) · record
- [official_result_331684.json](official_result_331684.json) · record
- [closeout_official_v1.json](closeout_official_v1.json) · record
- [PACKING_PROTOCOL.md](PACKING_PROTOCOL.md) · config
- [official_completion_331684.json](official_completion_331684.json) · result
- [official_evaluation_331684.json](official_evaluation_331684.json) · result
- [official_comparison_331682_331684.json](../331682/official_comparison_331682_331684.json) · result
- [official_objective_gap_331684.json](official_objective_gap_331684.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- ← enables: [Standing approval for all qualified Phase 2 uploads recorded 2026-09-20 after the exact-archive approval of #331672](../../records/research/2026-09-20-standing-approval-all-qualified-phase-2-uploads-recorded-2026/README.md)
- ← shipped in: [Standing approval for all qualified Phase 2 uploads recorded 2026-09-20 after the exact-archive approval of #331672](../../records/research/2026-09-20-standing-approval-all-qualified-phase-2-uploads-recorded-2026/README.md)
- ← enables: [Copying 65 readout coefficients onto the covariance-edge estimator gives eight-network adjusted ratio 0.995858110143](../../experiments/learned-correction/copying-65-readout-coefficients-onto-covariance-edge-estimator-gives/README.md)
- measured by: [Frozen covariance-edge correction passes panel14 at ratio 0.982838708008 and confirmation8 at 0.982181085851](../../experiments/covariance-structure/frozen-covariance-edge-correction-passes-panel14-ratio-0-982838708008/README.md)
- supersedes: [Submission #331682: lossless static parameter packing of the 65-coefficient final-head refit; 100/100 completed](../331682/README.md)
- ← supersedes: [Submission #331759: copied 65-coefficient readout on the covariance-edge estimator; 11 of 100 networks failed](../331759/README.md)
- ← supersedes: [Submission #331760: 45200 mean-encoder parameters refit on the covariance-edge estimator; score 3.507738856975084e-9](../331760/README.md)

[Research graph](../../graph/index.html#node=submission-331684) · [Copy provenance](../../PROVENANCE.json)

