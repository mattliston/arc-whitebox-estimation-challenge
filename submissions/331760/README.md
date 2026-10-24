# Submission #331760: 45200 mean-encoder parameters refit on the covariance-edge estimator; score 3.507738856975084e-9

2026-09-21T15:13:48Z · submission · adopted · measured

The 11 existing keys of the learned mean-encoder P (the message-passing network that corrects the per-layer means: message weights, three encoder layers, head and gain, 45,200 parameters) were refit on the covariance-edge bank #331684 with everything else frozen (the birth-node matrix A, the per-layer cstar coefficients, the 528 covariance-edge values, normalizers). The archive submission-current-p-20260921.tar.gz (352307 bytes, SHA 81f4d31e...) was accepted at 2026-09-21T15:13:48.553542Z and completed 100 of 100 networks with zero failures at 2026-09-21T15:50:02Z, becoming the new official bank. Local evidence before the fire: native40 adjusted ratio 0.9848676992236052 (32/40 wins), cold14 0.9787744692559065 (12/14), confirmation8 0.9853436418458571 (7/8), independent noise-corrected public8 0.9719722866477218 and virgin6 0.9899910644606413.

Recorded measurement:

    official adjusted 3.507738856975084e-09, raw final-layer MSE 1.909537328259603e-08, 100/100 completed, 0 failed; relative adjusted improvement 0.02139137812873715 and raw improvement 0.020607167807568327 vs #331684 (3.5844144212297072e-09); telemetry flops_used min/mean/max 380333052905.0 / 400721384195.17 / 422449071076.0, participant predict wall 93.93084736300011 / 101.93889349724 / 113.31461067900001 s, residual 0.29678166699999053 / 0.31830606403999995 / 0.3824777899999958 s; local release: pooled cold wall ratio 0.999551747, worst 1.001999837, max residual 0.204172734 s, peak RSS 7030669312 bytes

Public score: 3.507738856975084e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [REEXAMINATION_INDEX_20260919.md#L433](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L433)
- [REEXAMINATION_INDEX_20260919.md#L441](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L441)
- [OFFICIAL_RESULT.md#L3](OFFICIAL_RESULT.md#L3)
- [OFFICIAL_RESULT.md#L10](OFFICIAL_RESULT.md#L10)
- [READY_RESULT.md#L9](READY_RESULT.md#L9)
- [bank_and_signed_fit_checkpoint103.md#L3](../331759/bank_and_signed_fit_checkpoint103.md#L3)
- [downstream_release_checkpoint101.md#L3](../../instruments/downstream-scorer-checks-cold14-confirmation8-independent-public8-plus-virgin6/downstream_release_checkpoint101.md#L3)

<details>
<summary>All result files and records (19)</summary>

- [official_best_331760.json](official_best_331760.json) · record
- [accepted.json](accepted.json) · record
- [official_completion_331760.json](official_completion_331760.json) · record
- [upload_closeout.json](../../records/research/2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/upload_closeout.json) · record
- [OFFICIAL_RESULT.md](OFFICIAL_RESULT.md) · record
- [READY_RESULT.md](READY_RESULT.md) · record
- [bank_and_signed_fit_checkpoint103.md](../331759/bank_and_signed_fit_checkpoint103.md) · record
- [approved_uploads_checkpoint102.md](../../records/research/2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/approved_uploads_checkpoint102.md) · record
- [REEXAMINATION_INDEX_20260919.md](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record
- [evaluation_331760_100.json](evaluation_331760_100.json) · result
- [official_completion_331760-a31cbbaf.json](official_completion_331760-a31cbbaf.json) · result
- [official_result_331760.json](official_result_331760.json) · result
- [official_best_331760-4be8885b.json](official_best_331760-4be8885b.json) · result
- [evaluation_331760_01.json](evaluation_331760_01.json) · result
- [evaluation_331760_02.json](evaluation_331760_02.json) · result
- [evaluation_331760_03.json](evaluation_331760_03.json) · result
- [evaluation_331760_04.json](evaluation_331760_04.json) · result
- [evaluation_331760_05.json](evaluation_331760_05.json) · result
- [downstream_release_checkpoint101.md](../../instruments/downstream-scorer-checks-cold14-confirmation8-independent-public8-plus-virgin6/downstream_release_checkpoint101.md) · record

</details>

## Related work

- ← shipped in: [Allocator relief at graph boundaries cuts the plain forward high-water from 6,793,330,688 to 2,965,045,248 bytes](../../records/research/2026-09-21-allocator-relief-graph-boundaries-cuts-plain-forward-high-water/README.md)
- ← enables: [Exact-archive upload approvals for the readout composition and the refit mean encoder: both accepted 2026-09-21T15:13Z](../../records/research/2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/README.md)
- ← shipped in: [Exact-archive upload approvals for the readout composition and the refit mean encoder: both accepted 2026-09-21T15:13Z](../../records/research/2026-09-21-exact-archive-upload-approvals-readout-composition-refit-mean-encoder/README.md)
- ← enables: [Refit mean encoder on the 40-network validation split: adjusted ratio 0.9848676992236052, 32/40 wins at equal compute](../../experiments/evaluation-methodology/refit-mean-encoder-40-network-validation-split-adjusted-ratio/README.md)
- ← shipped in: [Refit mean encoder on the 40-network validation split: adjusted ratio 0.9848676992236052, 32/40 wins at equal compute](../../experiments/evaluation-methodology/refit-mean-encoder-40-network-validation-split-adjusted-ratio/README.md)
- ← enables: [Refit mean encoder passed all downstream gates: cold14 ratio 0.9787744692559065, independent public8 0.9719722866477218](../../experiments/evaluation-methodology/refit-mean-encoder-passed-all-downstream-gates-cold14-ratio/README.md)
- ← shipped in: [Refit mean encoder passed all downstream gates: cold14 ratio 0.9787744692559065, independent public8 0.9719722866477218](../../experiments/evaluation-methodology/refit-mean-encoder-passed-all-downstream-gates-cold14-ratio/README.md)
- ← shipped in: [Refitting 45200 mean-encoder parameters on the covariance-edge estimator completes 160 graphs and 40 updates](../../experiments/learned-correction/refitting-45200-mean-encoder-parameters-covariance-edge-estimator-completes/README.md)
- shipped in: [Refitting 45200 mean-encoder parameters on the covariance-edge estimator completes 160 graphs and 40 updates](../../experiments/learned-correction/refitting-45200-mean-encoder-parameters-covariance-edge-estimator-completes/README.md)
- supersedes: [Submission #331684: packed frozen covariance-edge correction at layers 8-14 with full lower-moment support](../331684/README.md)
- ← supersedes: [Submission #331824: 1,536-value message-to-covariance context fit improves adjusted error 0.018%](../331824/README.md)

[Research graph](../../graph/index.html#node=submission-331760) · [Copy provenance](../../PROVENANCE.json)

