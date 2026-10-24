# Submission #331759: copied 65-coefficient readout on the covariance-edge estimator; 11 of 100 networks failed

2026-09-21T15:13:28Z · submission · failed · measured

The fixed composition copied exactly the 64 final-head weights and the last bias (65 FP32 values) already served by the packed readout submission #331682 into the covariance-edge bank #331684, with no fit; locally it passed an eight-network gate (pooled adjusted/bank 0.995858110143, 5/8 wins) and the ordinary client checks (max candidate residual 0.209131764 s). The archive submission-covariance-readout-20260921.tar.gz (352353 bytes, SHA e9eae243...) was accepted at 2026-09-21T15:13:28.432327Z. Grading was failure-penalised: 89 networks succeeded and 11 failed, 4 by residual wall time (network 4 nicholas-gonzalez at 0.4047110810000021 s against the 0.4 s cap, with predict wall 113.493815336 s) and 7 by total time, so it is not eligible as a bank and must not be retried unchanged.

Recorded measurement:

    official adjusted 0.020395830787829743, raw 0.020395846325135702; successful 89, failed 11 (RESIDUAL_WALL_TIME_EXHAUSTED 4, TIME_EXHAUSTED 7); local 8-net gate ratio 0.995858110143, 5/8 wins, worst 1.004219075102, parent pooled 3.6242291716621936e-09 vs composition 3.609218013617435e-09; local client pooled wall ratio 1.001256078524, worst pair 1.016428910214, max residual 0.209131764 s

Public score: 0.020395830787829743. Status: GRADED. Failed networks: 11.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [REEXAMINATION_INDEX_20260919.md#L434](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L434)
- [REEXAMINATION_INDEX_20260919.md#L442](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L442)
- [bank_and_signed_fit_checkpoint103.md#L9](bank_and_signed_fit_checkpoint103.md#L9)
- [approved_uploads_checkpoint102.md#L16](../../records/research/2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/approved_uploads_checkpoint102.md#L16)
- [READY_RESULT.md#L15](../../experiments/learned-correction/copying-65-readout-coefficients-onto-covariance-edge-estimator-gives/READY_RESULT.md#L15)
- [READY_RESULT.md#L22](../../experiments/learned-correction/copying-65-readout-coefficients-onto-covariance-edge-estimator-gives/READY_RESULT.md#L22)
- [READY_RESULT.md#L36](../../experiments/learned-correction/copying-65-readout-coefficients-onto-covariance-edge-estimator-gives/READY_RESULT.md#L36)

<details>
<summary>All result files and records (12)</summary>

- [accepted.json](accepted.json) · record
- [official_completion_331759.json](official_completion_331759.json) · record
- [upload_closeout.json](../../records/research/2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/upload_closeout.json) · record
- [READY_RESULT.md](../../experiments/learned-correction/copying-65-readout-coefficients-onto-covariance-edge-estimator-gives/READY_RESULT.md) · record
- [ACCURACY_RESULT.md](../../experiments/learned-correction/copying-65-readout-coefficients-onto-covariance-edge-estimator-gives/ACCURACY_RESULT.md) · result
- [bank_and_signed_fit_checkpoint103.md](bank_and_signed_fit_checkpoint103.md) · record
- [approved_uploads_checkpoint102.md](../../records/research/2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/approved_uploads_checkpoint102.md) · record
- [evaluation_331759_01.json](evaluation_331759_01.json) · result
- [evaluation_331759_02.json](evaluation_331759_02.json) · result
- [evaluation_331759_03.json](evaluation_331759_03.json) · result
- [evaluation_331759_04.json](evaluation_331759_04.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- ← shipped in: [Copying 65 readout coefficients onto the covariance-edge estimator gives eight-network adjusted ratio 0.995858110143](../../experiments/learned-correction/copying-65-readout-coefficients-onto-covariance-edge-estimator-gives/README.md)
- ← enables: [Exact-archive upload approvals for the readout composition and the refit mean encoder: both accepted 2026-09-21T15:13Z](../../records/research/2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/README.md)
- ← shipped in: [Exact-archive upload approvals for the readout composition and the refit mean encoder: both accepted 2026-09-21T15:13Z](../../records/research/2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/README.md)
- ← measured by: [Local residual margins do not establish remote reliability: 0.209 s local maximum, 0.4047 s remote failure at 0.4 s cap](../../records/research/2026-09-21-local-residual-margins-do-not-establish-remote-reliability-0/README.md)
- shipped in: [Copying 65 readout coefficients onto the covariance-edge estimator gives eight-network adjusted ratio 0.995858110143](../../experiments/learned-correction/copying-65-readout-coefficients-onto-covariance-edge-estimator-gives/README.md)
- enables: [Local residual margins do not establish remote reliability: 0.209 s local maximum, 0.4047 s remote failure at 0.4 s cap](../../records/research/2026-09-21-local-residual-margins-do-not-establish-remote-reliability-0/README.md)
- enables: [Submission #331682: lossless static parameter packing of the 65-coefficient final-head refit; 100/100 completed](../331682/README.md)
- supersedes: [Submission #331684: packed frozen covariance-edge correction at layers 8-14 with full lower-moment support](../331684/README.md)

[Research graph](../../graph/index.html#node=submission-331759) · [Copy provenance](../../PROVENANCE.json)

