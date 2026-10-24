# Submission #332207: alternative-basis Winograd arithmetic

2026-09-24T22:37:33Z · submission · adopted · measured

Replaced the fast matrix-product basis with the Karstadt–Schwartz alternative basis, reducing additions while preserving accuracy. The official public score is 2.9042030546701915e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332207, "date": "2026-09-24T22:37:33Z", "status": "GRADED", "score_public": 2.9042030546701915e-09, "n": 100, "failed_networks": 0, "utilization": 0.16396360117002587, "wall_mean_s": 64.642, "wall_max_s": 69.532, "resid_mean_s": 0.258, "resid_max_s": 0.309, "source_sha256": "e313ab775de8c0995f8e9920151021ad1992cb9b6979d95dd1daa9bd74f5154c"}
    - 2026-09-24 15:38 PDT FIRE #332207 (Fable): fable_fire_v1/fire_candS15/candS15_S14_altbasis_mem.py (sha256 e313ab77...) = Fable S15 = S14 #332203 + engineer2's --altbasis (Karstadt-Schwartz alternative-basis Winograd: 12 adds/level + 1 basis add per level per operand, on the tree kernel).
    - 2026-09-24 ~16:15 PDT GRADED #332207 (S15): adjusted 2.9042030546701915e-9, raw 1.772250367793049e-8, util 0.16396. NEW PHASE-2 BANK, -1.21% vs #332203 (2.9398e-9); expected 2.905e-9.
    - 2026-09-24 16:14 PDT FIRE #332210 (Fable): fable_fire_v1/fire_candS16/candS16_S15_bb6_mem.py (sha256 a4a1e0f7...) = Fable S16 = S15 #332207 + bb6 (the batched 1024^3 covariance/star products at Winograd depth 6 through the alt-basis batched path; engineer E-59, verified bit-exact against engineer2's independent run).
    - 2026-09-24 ~16:50 PDT GRADED #332210 (S16): adjusted 2.8603700803210566e-9, raw 1.7723716290163338e-8, util 0.16148. NEW PHASE-2 BANK, -1.51% vs #332207 (2.9042e-9); expected 2.863e-9.

Public score: 2.9042030546701915e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L762](../../records/SUBMISSION_LEDGER.md#L762)
- [SUBMISSION_LEDGER.md#L769](../../records/SUBMISSION_LEDGER.md#L769)
- [SUBMISSION_LEDGER.md#L773](../../records/SUBMISSION_LEDGER.md#L773)
- [SUBMISSION_LEDGER.md#L779](../../records/SUBMISSION_LEDGER.md#L779)
- [TEAM_LOG_2026-09-22_to_09-26.md#L448](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L448)
- [TEAM_LOG_2026-09-22_to_09-26.md#L454](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L454)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332207.json](eval_332207.json) · result

</details>

## Related work

- ← shipped in: [Alternative-basis Winograd reduces metered work 1.22% and user CPU 3.29%](../../experiments/arithmetic-kernels/alternative-basis-winograd-reduces-metered-work-1-22-user/README.md)
- supersedes: [Submission #332203: tree-based six-level reads, row padding, and general depth selection](../332203/README.md)
- ← supersedes: [Submission #332210: batched six-level covariance and residual-plane products](../332210/README.md)

[Research graph](../../graph/index.html#node=submission-332207) · [Copy provenance](../../PROVENANCE.json)

