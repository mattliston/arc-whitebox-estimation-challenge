# Submission #331672: full lower-moment support (full covariance, star and trace reads) added to the lazy-rotation parent

2026-09-20T17:53:09Z · submission · adopted · measured

The candidate keeps the full covariance, the repeated third-moment coordinate-star target and the trace vector b with natural-order full reads on the unchanged lazy coefficient-rotation parent (#331599, called Lazy), with CP masks, births, both 0.01 thresholds, rank 384, age 4 and every learned parameter fixed. It was accepted on 2026-09-20T17:53:09.338Z under the owner's explicit approval of the exact archive and graded adjusted 3.6316074532884024e-9, raw 1.9930081087693453e-8, the new official best. Its strict development gate (>=5% adjusted gain on each of nets 35 and 100) had failed with a pooled 2.1668% gain, but the broader 14-net noise-corrected panel and the 8-net confirmation passed, which justified one platform trial.

Recorded measurement:

    official adjusted 3.6316074532884024e-9, raw 1.9930081087693453e-8; 4.843293% better adjusted and 8.125170% better raw than #331599 (3.8164492657795035e-9); local NC14 adjusted 3.688965748863211e-9 ratio 0.966649339555 (9/14 wins); confirmation8 ordinary adjusted 3.689980619531317e-9 ratio 0.956240220372 (8/8); archive 779737 bytes sha256 01e05883c1767534e23b2f4ee0d0e4b887f777c172080f14b3a91d70ea3402fd; estimator sha256 bbae60ddfad2718da44d4bcc6cf50110998675fcc5f7a79e6bc4dd7c5b4b9d12

Public score: 3.6316074532884024e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [OFFICIAL_RESULT.md#L6](OFFICIAL_RESULT.md#L6)
- [OFFICIAL_RESULT.md#L7](OFFICIAL_RESULT.md#L7)
- [OFFICIAL_RESULT.md#L8](OFFICIAL_RESULT.md#L8)
- [RESULT_v2.md#L5](../../experiments/covariance-structure/full-lower-moment-support-passes-broader-panel-adjusted-ratio/RESULT_v2.md#L5)
- [RESULT_v2.md#L9](../../experiments/covariance-structure/full-lower-moment-support-passes-broader-panel-adjusted-ratio/RESULT_v2.md#L9)
- [RESULT_v2.md#L11](../../experiments/covariance-structure/full-lower-moment-support-passes-broader-panel-adjusted-ratio/RESULT_v2.md#L11)
- [RESULT_v2.md#L13](../../experiments/covariance-structure/full-lower-moment-support-passes-broader-panel-adjusted-ratio/RESULT_v2.md#L13)
- [REEXAMINATION_INDEX_20260919.md#L889](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L889)
- [REEXAMINATION_INDEX_20260919.md#L893](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L893)
- [REEXAMINATION_INDEX_20260919.md#L880](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L880)
- [REEXAMINATION_INDEX_20260919.md#L882](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L882)

<details>
<summary>All result files and records (8)</summary>

- [OFFICIAL_RESULT.md](OFFICIAL_RESULT.md) · record
- [RESULT_v2.md](../../experiments/covariance-structure/full-lower-moment-support-passes-broader-panel-adjusted-ratio/RESULT_v2.md) · record
- [official_best_331672.json](official_best_331672.json) · record
- [closeout_after_submission_v1.json](closeout_after_submission_v1.json) · record
- [submission.json](submission.json) · record
- [PROTOCOL.md](../../experiments/covariance-structure/full-lower-moment-support-passes-broader-panel-adjusted-ratio/PROTOCOL.md) · config
- [official_objective_gap_331672.json](official_objective_gap_331672.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- ← enables: [Full lower-moment support on nets 35/100: adjusted ratios 0.99071/0.96585 fail the 5%-each gate (pooled 2.1668% better)](../../experiments/covariance-structure/full-lower-moment-support-nets-35-100-adjusted-ratios/README.md)
- ← enables: [Standing approval for all qualified Phase 2 uploads recorded 2026-09-20 after the exact-archive approval of #331672](../../records/research/2026-09-20-standing-approval-all-qualified-phase-2-uploads-recorded-2026/README.md)
- ← shipped in: [Standing approval for all qualified Phase 2 uploads recorded 2026-09-20 after the exact-archive approval of #331672](../../records/research/2026-09-20-standing-approval-all-qualified-phase-2-uploads-recorded-2026/README.md)
- measured by: [Full lower-moment support on nets 35/100: adjusted ratios 0.99071/0.96585 fail the 5%-each gate (pooled 2.1668% better)](../../experiments/covariance-structure/full-lower-moment-support-nets-35-100-adjusted-ratios/README.md)
- enables: [Standing approval for all qualified Phase 2 uploads recorded 2026-09-20 after the exact-archive approval of #331672](../../records/research/2026-09-20-standing-approval-all-qualified-phase-2-uploads-recorded-2026/README.md)
- supersedes: [Submission #331599: delay four old-tier coefficient rotations, official score 3.816449265779503e-09](../331599/README.md)
- ← supersedes: [Submission #331676: 65-literal final-head ridge refit of #331672 failed: two networks exceeded the 0.4 s residual cap](../331676/README.md)

[Research graph](../../graph/index.html#node=submission-331672) · [Copy provenance](../../PROVENANCE.json)

