# Submission #336075: feasible head mean, dense covariance reuse and Gaussian T22 reuse on #335920, 1.51317e-9

2026-10-14T18:16:10Z · submission · adopted · measured

The feasible head mean (one fixed unit d_eff = max(dm, -m) at layers 1 to 14) was stacked with two exact compute reuses (dense covariance reuse and the Gaussian T22 common product) on the #335920 file, with inverse output pruning disabled after the request gate failure. It became the new public best by 0.09%. The lead fired it under the owner's three-gate rule (seal negative, release confirmed, prefire clear) although the ten-network native diagnostic was flat.

Recorded measurement:

    official 1.5131679660885974e-9 (new best), -0.089100% vs #335920; paired 50 meanlog -0.092250% (SE 0.045236%, 27/50 lower); final MSE -0.020%, bill log -0.075%; residual max 0.380 s, setup 4.398 s; seal 8 meanlog net -0.117412% (SE 0.091914%); canonical native ten +0.005339% (SE 0.094635%); normal production risk 15.04%

Public score: 1.5131679660885974e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2957](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2957)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2961](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2961)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2991](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2991)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2993](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2993)
- [SUBMISSION_LEDGER.md#L971](../../records/SUBMISSION_LEDGER.md#L971)
- [SUBMISSION_LEDGER.md#L972](../../records/SUBMISSION_LEDGER.md#L972)

<details>
<summary>All result files and records (9)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [attempt.json](attempt.json) · record
- [eval_336075.json](eval_336075.json) · result
- [stage_a.json](stage_a.json) · record
- [ROOT_PUBLIC_GRADE_v1.json](ROOT_PUBLIC_GRADE_v1.json) · result
- [ROOT_FIRE_DECISION_v1.json](ROOT_FIRE_DECISION_v1.json) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Feasible head mean d_eff = max(dm, -m) at layers 1-14 before covariance compensation: fresh -0.076%, OOS -0.075%](../../experiments/moment-propagation/feasible-head-mean-d-eff-max-dm-m-layers/README.md)
- ← shipped in: [Gaussian second-order common-product reuse: exact, 64,684,032 FLOPs saved for 140 additional requests](../../experiments/flop-accounting/gaussian-second-order-common-product-reuse-exact-64-684/README.md)
- ← shipped in: [Joint Gaussian T22 reuse plus inverse output pruning: exact, bill -0.069%/-0.059% (mean -0.064%), requests +2,119/+2,452](../../experiments/flop-accounting/joint-gaussian-t22-reuse-plus-inverse-output-pruning-exact/README.md)
- ← enables: [Stack with inverse output pruning fails the request gate (loss risk 88.33%, 57,693/52,643 requests); pruning dropped](../../records/research/2026-10-14-stack-inverse-output-pruning-fails-request-gate-loss-risk/README.md)
- ← shipped in: [Feasible mean and rank-two output projection absorbed into refit rounds 61 and 62](../../records/research/2026-10-15-feasible-mean-rank-two-output-projection-absorbed-into-refit/README.md)
- ← enables: [Submission #335920: bank-product route fusion, fresh-buffer copying and quarter input loading, bill -0.273%](../335920/README.md)
- shipped in: [Feasible head mean d_eff = max(dm, -m) at layers 1-14 before covariance compensation: fresh -0.076%, OOS -0.075%](../../experiments/moment-propagation/feasible-head-mean-d-eff-max-dm-m-layers/README.md)
- enables: [Submission #336101: rank-two learned output projection with planner request repair, public score 1.50514e-9](../336101/README.md)

[Research graph](../../graph/index.html#node=submission-336075) · [Copy provenance](../../PROVENANCE.json)

