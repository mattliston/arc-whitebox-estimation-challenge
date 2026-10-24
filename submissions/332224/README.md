# Submission #332224: half-strength unit birth feed, refit, lean views, and setup prebinding

2026-09-25T01:34:05Z · submission · adopted · measured

Added half-strength per-unit birth feed with refitted closure parameters, leaner views, and setup prebinding. The official public score is 2.818244042956051e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332224, "date": "2026-09-25T01:34:05Z", "status": "GRADED", "score_public": 2.818244042956051e-09, "n": 100, "failed_networks": 0, "utilization": 0.16103773140161137, "wall_mean_s": 67.692, "wall_max_s": 73.326, "resid_mean_s": 0.24, "resid_max_s": 0.267, "source_sha256": "9a4ccc1a845b1b6a334fdfe4ec555826ced98a7dd8f664128fb4b1d9f081397f"}
    - 2026-09-24 18:36 PDT FIRE #332224 (Fable): fable_fire_v1/fire_candS17/candS17_S16r3bp2b_j05_mem.py (sha256 9a4ccc1a...) = Fable S17 = S16 #332210 plus:
    - 2026-09-24 ~19:15 PDT GRADED #332224 (S17): adjusted 2.818244042956051e-9, raw 1.7507960343721153e-8, util 0.16104. NEW PHASE-2 BANK, -1.47% vs #332210 (2.8604e-9); expected ~2.79e-9.
    - 2026-09-24 19:32 PDT FIRE #332226 (Fable): fable_fire_v1/fire_candS18/candS18_S17_lrq1r32_mem.py (sha256 ab8c8ba8...) = Fable S18 = S17 #332224 + low-rank star plane.
    - 2026-09-24 ~19:25 PDT GRADED #332226 (S18): adjusted 2.670488110920038e-9, raw 1.7577477251506934e-8, util 0.15199. NEW PHASE-2 BANK, -5.24% vs #332224 (2.8182e-9); expected ~2.667e-9.

Public score: 2.818244042956051e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L783](../../records/SUBMISSION_LEDGER.md#L783)
- [SUBMISSION_LEDGER.md#L792](../../records/SUBMISSION_LEDGER.md#L792)
- [SUBMISSION_LEDGER.md#L797](../../records/SUBMISSION_LEDGER.md#L797)
- [SUBMISSION_LEDGER.md#L805](../../records/SUBMISSION_LEDGER.md#L805)
- [TEAM_LOG_2026-09-22_to_09-26.md#L554](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L554)
- [TEAM_LOG_2026-09-22_to_09-26.md#L563](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L563)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332224.json](eval_332224.json) · result

</details>

## Related work

- ← shipped in: [Born-unit plane-to-distinct feed at half strength improves learned-off raw error 6.3%](../../experiments/birth-terms/born-unit-plane-distinct-feed-half-strength-improves-learned/README.md)
- ← shipped in: [Half-strength born-unit plane feed plus refit improves 40-network raw error 2.05%](../../experiments/learned-correction/half-strength-born-unit-plane-feed-plus-refit-improves/README.md)
- ← shipped in: [Natural-order tree views remove 1,455-1,565 calls with identical outputs, work and peak live memory](../../experiments/arithmetic-kernels/natural-order-tree-views-remove-1-455-1-565/README.md)
- ← shipped in: [Repeated-shape setup prebinding saves 1.7% cold work; all five platform workers complete setup in about 2 seconds](../../experiments/setup-budget/repeated-shape-setup-prebinding-saves-1-7-cold-work/README.md)
- supersedes: [Submission #332210: batched six-level covariance and residual-plane products](../332210/README.md)
- ← supersedes: [Submission #332226: rank-32 memoryless (2,1) residual plane with one power iteration](../332226/README.md)

[Research graph](../../graph/index.html#node=submission-332224) · [Copy provenance](../../PROVENANCE.json)

