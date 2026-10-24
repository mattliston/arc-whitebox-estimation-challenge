# Submission #328586: depth-1 third-cumulant Winograd arena reduced compute with unchanged raw error

2026-08-26T23:31:48Z · submission · adopted · measured

The depth-1 arena preserves output to float32 roundoff while saving about 118.5B hot third-cumulant canonical-polyadic carrier (CP3) FLOPs. Its local A/B/A gate failed only the final rank35 residual by 0.00020734211429954 s, but the official grader is authoritative: submission #328586 graded 100/100 at adjusted 2.1258043062525995e-8 and raw 2.9409214512554626e-8, improving #328439 adjusted by 6.94% with raw effectively unchanged.

Recorded measurement:

    The depth-1 arena preserves output to float32 roundoff while saving about 118.5B hot CP3 FLOPs. Its local A/B/A gate failed only the final rank35 residual by 0.00020734211429954 s, but the official grader is authoritative: submission #328586 graded 100/100 at adjusted 2.1258043062525995e-8 and raw 2.9409214512554626e-8, improving #328439 adjusted by 6.94% with raw effectively unchanged. Official public record: {"status": "GRADED", "score_public": 2.1258043062525995e-08, "n": 100, "failed_networks": 0, "utilization": 0.7227992452633316, "wall_mean_s": 45.446, "wall_max_s": 53.253, "resid_mean_s": 0.101, "resid_max_s": 0.109}

Public score: 2.1258043062525995e-08. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [D1_CANDIDATE_RESULT.json#L1](../../experiments/arithmetic-kernels/depth-1-third-cumulant-arena-deployed-submission-328586/D1_CANDIDATE_RESULT.json#L1)
- [SUBMIT_WATCH.log#L1](../../experiments/flop-accounting/deployed-estimator-score-economics-minimum-charged-utilisation-verified/SUBMIT_WATCH.log#L1)

<details>
<summary>All result files and records (3)</summary>

- [D1_CANDIDATE_RESULT.json](../../experiments/arithmetic-kernels/depth-1-third-cumulant-arena-deployed-submission-328586/D1_CANDIDATE_RESULT.json) · result
- [SUBMIT_WATCH.log](../../experiments/flop-accounting/deployed-estimator-score-economics-minimum-charged-utilisation-verified/SUBMIT_WATCH.log) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

[Research graph](../../graph/index.html#node=submission-328586) · [Copy provenance](../../PROVENANCE.json)

