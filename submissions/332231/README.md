# Submission #332231: epoch-2 learned parameters with first-round physics scales

2026-09-25T03:56:27Z · submission · adopted · measured

Replaced learned literals with the epoch-2 refit while retaining first-round physics scales and the additional six-level products. The official public score is 2.5832095956918383e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332231, "date": "2026-09-25T03:56:27Z", "status": "GRADED", "score_public": 2.5832095956918383e-09, "n": 100, "failed_networks": 0, "utilization": 0.15119250627920336, "wall_mean_s": 66.694, "wall_max_s": 70.924, "resid_mean_s": 0.242, "resid_max_s": 0.281, "source_sha256": "eddbcfcdb164185a1b80d7a592e06bb6b5df2e897ecf7a413f790ddad0e51b94"}
    - 2026-09-24 20:56 PDT FIRE #332231 (Fable): fable_fire_v1/fire_candS21/candS21_S18ep2_dials_d6x_mem.py (sha256 eddbcfcd...) = Fable S21 = S18 #332226 + physicist's j05 ep2 refit literals (28 literals bit-equal to params) + covhunter's round-1 shared-semantics dials (c0 .709, c1 .503, k4 .989/.963/.960, X1 .559, X2 .358, X3 .152, b 1.051, lam 1.0047/.9931) + engineer2's d6x. Built by the porter's chain.
    - 2026-09-24 21:21 PDT GRADED #332231 (S21): adjusted 2.5832095956918383e-9, raw 1.7088557644484582e-8, util 0.15119. NEW PHASE-2 BANK, -0.40% vs #332230 (2.5937e-9), -1.40% vs S19; expected ~2.58e-9 (VALID40 vs S19 -1.46%).
    - 2026-09-24 22:07 PDT FIRE #332236 (Fable): fable_fire_v1/fire_candS21s1/candS21s1_S21_s1lr_mem.py (sha256 30aea66e...) = Fable S21s1 = S21 #332231 + engineer2's low-rank s1 (the star's dense product s1 = (push o w^T) w computed from pw's rank-32 factors; C w as one depth-6 product). Porter staged: composition byte-exact, recipe rebuild bit-equal (28 literals).
    - 2026-09-24 22:32 PDT GRADED #332236 (S21s1): adjusted 2.533185551983547e-9, raw 1.7103584220024003e-8, util 0.14813. NEW PHASE-2 BANK, -1.94% vs #332231 (2.5832e-9); expected ~2.537e-9 (VALID40 -1.78%).

Public score: 2.5832095956918383e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L829](../../records/SUBMISSION_LEDGER.md#L829)
- [SUBMISSION_LEDGER.md#L839](../../records/SUBMISSION_LEDGER.md#L839)
- [SUBMISSION_LEDGER.md#L844](../../records/SUBMISSION_LEDGER.md#L844)
- [SUBMISSION_LEDGER.md#L850](../../records/SUBMISSION_LEDGER.md#L850)
- [TEAM_LOG_2026-09-22_to_09-26.md#L622](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L622)
- [TEAM_LOG_2026-09-22_to_09-26.md#L625](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L625)
- [TEAM_LOG_2026-09-22_to_09-26.md#L647](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L647)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332231.json](eval_332231.json) · result

</details>

## Related work

- ← shipped in: [Alternating learned-on dial fit and refit improves held-out error 0.89% in the first round](../../experiments/dials/alternating-learned-dial-fit-refit-improves-held-out-error/README.md)
- ← shipped in: [Second birth-feed refit epoch improves sixteen-network validation error 0.78% with 16/16 wins](../../experiments/learned-correction/second-birth-feed-refit-epoch-improves-sixteen-network-validation/README.md)
- ← shipped in: [Second physical-feed refit epoch improves all 40 held-out networks by 0.69%](../../experiments/learned-correction/second-physical-feed-refit-epoch-improves-all-40-held/README.md)
- supersedes: [Submission #332230: refitted physics scales and additional six-level products](../332230/README.md)
- ← supersedes: [Submission #332236: rank-32 evaluation of the residual-plane cross term](../332236/README.md)

[Research graph](../../graph/index.html#node=submission-332231) · [Copy provenance](../../PROVENANCE.json)

