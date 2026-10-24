# Submission #332236: rank-32 evaluation of the residual-plane cross term

2026-09-25T05:07:52Z · submission · adopted · measured

Evaluated the memoryless residual-plane cross term from rank-32 factors and used a six-level covariance product. The official public score is 2.533185551983547e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332236, "date": "2026-09-25T05:07:52Z", "status": "GRADED", "score_public": 2.533185551983547e-09, "n": 100, "failed_networks": 0, "utilization": 0.14813382883469786, "wall_mean_s": 64.606, "wall_max_s": 70.168, "resid_mean_s": 0.239, "resid_max_s": 0.254, "source_sha256": "30aea66ef5afa24a10a6f543e881eac3c36d9c24e5a6996c61ee5123e693c970"}
    - 2026-09-24 22:07 PDT FIRE #332236 (Fable): fable_fire_v1/fire_candS21s1/candS21s1_S21_s1lr_mem.py (sha256 30aea66e...) = Fable S21s1 = S21 #332231 + engineer2's low-rank s1 (the star's dense product s1 = (push o w^T) w computed from pw's rank-32 factors; C w as one depth-6 product). Porter staged: composition byte-exact, recipe rebuild bit-equal (28 literals).
    - 2026-09-24 22:32 PDT GRADED #332236 (S21s1): adjusted 2.533185551983547e-9, raw 1.7103584220024003e-8, util 0.14813. NEW PHASE-2 BANK, -1.94% vs #332231 (2.5832e-9); expected ~2.537e-9 (VALID40 -1.78%).
    - 2026-09-24 22:46 PDT FIRE #332241 (Fable): fable_fire_v1/fire_candS22/candS22_S21s1_r2_mem.py (sha256 33d4f0d0...) = Fable S22 = S21s1 #332236 + the physicist's loopR1 refit literals (one refit epoch on the round-1-dial card, warm from j05 ep2; VALID16 -0.95%) + round-1 dials unchanged (the round-2 dial fit converged: LOO +0.44%).
    - 2026-09-24 23:11 PDT GRADED #332241 (S22): adjusted 2.5054244494808718e-9, raw 1.6916191079019428e-8, util 0.14815. NEW PHASE-2 BANK, -1.10% vs #332236 (2.5332e-9); expected ~2.501e-9 (VALID40 -1.26%).

Public score: 2.533185551983547e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L844](../../records/SUBMISSION_LEDGER.md#L844)
- [SUBMISSION_LEDGER.md#L850](../../records/SUBMISSION_LEDGER.md#L850)
- [SUBMISSION_LEDGER.md#L854](../../records/SUBMISSION_LEDGER.md#L854)
- [SUBMISSION_LEDGER.md#L860](../../records/SUBMISSION_LEDGER.md#L860)
- [TEAM_LOG_2026-09-22_to_09-26.md#L689](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L689)
- [TEAM_LOG_2026-09-22_to_09-26.md#L711](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L711)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332236.json](eval_332236.json) · result

</details>

## Related work

- ← shipped in: [Rank-32 residual-plane cross term improves adjusted error 1.782% on forty networks](../../experiments/third-order-planes/rank-32-residual-plane-cross-term-improves-adjusted-error/README.md)
- supersedes: [Submission #332231: epoch-2 learned parameters with first-round physics scales](../332231/README.md)
- ← supersedes: [Submission #332241: one-epoch learned refit at the fitted physics scales](../332241/README.md)

[Research graph](../../graph/index.html#node=submission-332236) · [Copy provenance](../../PROVENANCE.json)

