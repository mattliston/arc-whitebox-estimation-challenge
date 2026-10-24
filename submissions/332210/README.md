# Submission #332210: batched six-level covariance and residual-plane products

2026-09-24T23:12:09Z · submission · adopted · measured

Moved the batched covariance and memoryless (2,1) residual-plane products to six levels of the alternative-basis kernel. The official public score is 2.8603700803210566e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332210, "date": "2026-09-24T23:12:09Z", "status": "GRADED", "score_public": 2.8603700803210566e-09, "n": 100, "failed_networks": 0, "utilization": 0.16147891916109983, "wall_mean_s": 69.505, "wall_max_s": 74.349, "resid_mean_s": 0.262, "resid_max_s": 0.313, "source_sha256": "a4a1e0f74a3100bd061a39d4e6ea1cde1b0271cd3746f4d64ff520d6e0458ab9"}
    - 2026-09-24 16:14 PDT FIRE #332210 (Fable): fable_fire_v1/fire_candS16/candS16_S15_bb6_mem.py (sha256 a4a1e0f7...) = Fable S16 = S15 #332207 + bb6 (the batched 1024^3 covariance/star products at Winograd depth 6 through the alt-basis batched path; engineer E-59, verified bit-exact against engineer2's independent run).
    - 2026-09-24 ~16:50 PDT GRADED #332210 (S16): adjusted 2.8603700803210566e-9, raw 1.7723716290163338e-8, util 0.16148. NEW PHASE-2 BANK, -1.51% vs #332207 (2.9042e-9); expected 2.863e-9.
    - 2026-09-24 18:36 PDT FIRE #332224 (Fable): fable_fire_v1/fire_candS17/candS17_S16r3bp2b_j05_mem.py (sha256 9a4ccc1a...) = Fable S17 = S16 #332210 plus:
    - 2026-09-24 ~19:15 PDT GRADED #332224 (S17): adjusted 2.818244042956051e-9, raw 1.7507960343721153e-8, util 0.16104. NEW PHASE-2 BANK, -1.47% vs #332210 (2.8604e-9); expected ~2.79e-9.

Public score: 2.8603700803210566e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L773](../../records/SUBMISSION_LEDGER.md#L773)
- [SUBMISSION_LEDGER.md#L779](../../records/SUBMISSION_LEDGER.md#L779)
- [SUBMISSION_LEDGER.md#L783](../../records/SUBMISSION_LEDGER.md#L783)
- [SUBMISSION_LEDGER.md#L792](../../records/SUBMISSION_LEDGER.md#L792)
- [TEAM_LOG_2026-09-22_to_09-26.md#L460](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L460)
- [TEAM_LOG_2026-09-22_to_09-26.md#L473](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L473)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332210.json](eval_332210.json) · result

</details>

## Related work

- ← shipped in: [Six-level batched covariance and residual-plane products improve four-network adjusted error 1.557%](../../experiments/arithmetic-kernels/six-level-batched-covariance-residual-plane-products-improve-four/README.md)
- supersedes: [Submission #332207: alternative-basis Winograd arithmetic](../332207/README.md)
- ← supersedes: [Submission #332224: half-strength unit birth feed, refit, lean views, and setup prebinding](../332224/README.md)

[Research graph](../../graph/index.html#node=submission-332210) · [Copy provenance](../../PROVENANCE.json)

