# Submission #332203: tree-based six-level reads, row padding, and general depth selection

2026-09-24T22:04:49Z · submission · adopted · measured

Added tree-based six-level reads, padding to 64-row blocks, and a general product-depth rule. The official public score is 2.939833791514085e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332203, "date": "2026-09-24T22:04:49Z", "status": "GRADED", "score_public": 2.939833791514085e-09, "n": 100, "failed_networks": 0, "utilization": 0.1659709788501368, "wall_mean_s": 67.791, "wall_max_s": 74.426, "resid_mean_s": 0.25, "resid_max_s": 0.276, "source_sha256": "9003e5368e7fc27849ba969056f3700fbac0206143297cae92762aa55f408b8f"}
    - 2026-09-24 15:05 PDT FIRE #332203 (Fable): fable_fire_v1/fire_candS14/candS14_S13_tree_r6_m64_mem.py (sha256 9003e5368e7fc27849ba969056f3700fbac0206143297cae92762aa55f408b8f) = Fable S14. Additions to S13 #332195 (engineer E-54..E-57):
    - 2026-09-24 ~15:30 PDT GRADED #332203 (S14): adjusted 2.939833791514085e-9, raw 1.772207095740441e-8, util 0.16597. NEW PHASE-2 BANK, -1.16% vs #332195 (2.9743e-9); expected ~2.946e-9.
    - 2026-09-24 15:38 PDT FIRE #332207 (Fable): fable_fire_v1/fire_candS15/candS15_S14_altbasis_mem.py (sha256 e313ab77...) = Fable S15 = S14 #332203 + engineer2's --altbasis (Karstadt-Schwartz alternative-basis Winograd: 12 adds/level + 1 basis add per level per operand, on the tree kernel).
    - 2026-09-24 ~16:15 PDT GRADED #332207 (S15): adjusted 2.9042030546701915e-9, raw 1.772250367793049e-8, util 0.16396. NEW PHASE-2 BANK, -1.21% vs #332203 (2.9398e-9); expected 2.905e-9.

Public score: 2.939833791514085e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L747](../../records/SUBMISSION_LEDGER.md#L747)
- [SUBMISSION_LEDGER.md#L757](../../records/SUBMISSION_LEDGER.md#L757)
- [SUBMISSION_LEDGER.md#L762](../../records/SUBMISSION_LEDGER.md#L762)
- [SUBMISSION_LEDGER.md#L769](../../records/SUBMISSION_LEDGER.md#L769)
- [TEAM_LOG_2026-09-22_to_09-26.md#L430](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L430)
- [TEAM_LOG_2026-09-22_to_09-26.md#L443](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L443)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332203.json](eval_332203.json) · result

</details>

## Related work

- ← shipped in: [Tree six-level reads and 64-row padding: adjusted error improves 0.948% on 40 networks](../../experiments/arithmetic-kernels/tree-six-level-reads-64-row-padding-adjusted-error/README.md)
- supersedes: [Submission #332195: column packing, inactive-row selection, six-level reads, and in-place tree products](../332195/README.md)
- ← supersedes: [Submission #332207: alternative-basis Winograd arithmetic](../332207/README.md)

[Research graph](../../graph/index.html#node=submission-332203) · [Copy provenance](../../PROVENANCE.json)

