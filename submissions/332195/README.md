# Submission #332195: column packing, inactive-row selection, six-level reads, and in-place tree products

2026-09-24T20:52:37Z · submission · adopted · measured

Added column packing, inactive-row selection, selected six-level reads, and an in-place tree implementation of fast matrix products. The official public score is 2.9742839646322686e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332195, "date": "2026-09-24T20:52:37Z", "status": "GRADED", "score_public": 2.9742839646322686e-09, "n": 100, "failed_networks": 0, "utilization": 0.16776196964578957, "wall_mean_s": 65.768, "wall_max_s": 72.85, "resid_mean_s": 0.258, "resid_max_s": 0.278, "source_sha256": "2a5dd324d78c4db6bc457485ce050a1e93d0ec2dea2654d0a6346b7de4119249"}
    - 2026-09-24 13:53 PDT FIRE #332195 (Fable): fable_fire_v1/fire_candS13/candS13_S12_ipt3_mem.py (sha256 2a5dd324d78c4db6bc457485ce050a1e93d0ec2dea2654d0a6346b7de4119249) = Fable S13, built on S11 #332182 in this order:
    - 2026-09-24 ~14:30 PDT GRADED #332195 (S13): adjusted 2.9742839646322686e-9, raw 1.773872847721236e-8, util 0.16776. NEW PHASE-2 BANK and FIRST SUB-3e-9 (the 3e-9 milestone), -2.37% vs #332182 (3.0463e-9).
    - 2026-09-24 15:05 PDT FIRE #332203 (Fable): fable_fire_v1/fire_candS14/candS14_S13_tree_r6_m64_mem.py (sha256 9003e5368e7fc27849ba969056f3700fbac0206143297cae92762aa55f408b8f) = Fable S14. Additions to S13 #332195 (engineer E-54..E-57):
    - 2026-09-24 ~15:30 PDT GRADED #332203 (S14): adjusted 2.939833791514085e-9, raw 1.772207095740441e-8, util 0.16597. NEW PHASE-2 BANK, -1.16% vs #332195 (2.9743e-9); expected ~2.946e-9.

Public score: 2.9742839646322686e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L732](../../records/SUBMISSION_LEDGER.md#L732)
- [SUBMISSION_LEDGER.md#L743](../../records/SUBMISSION_LEDGER.md#L743)
- [SUBMISSION_LEDGER.md#L747](../../records/SUBMISSION_LEDGER.md#L747)
- [SUBMISSION_LEDGER.md#L757](../../records/SUBMISSION_LEDGER.md#L757)
- [TEAM_LOG_2026-09-22_to_09-26.md#L402](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L402)
- [TEAM_LOG_2026-09-22_to_09-26.md#L406](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L406)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332195.json](eval_332195.json) · result

</details>

## Related work

- ← shipped in: [Dead-side column cutoff at -3.5 preserves raw error but native kernels erase most predicted savings](../../experiments/third-order-planes/dead-side-column-cutoff-3-5-preserves-raw-error/README.md)
- ← shipped in: [Packetized six-level reads on eligible shapes save 0.363% work with rounding-level error](../../experiments/arithmetic-kernels/packetized-six-level-reads-eligible-shapes-save-0-363/README.md)
- ← shipped in: [In-place tree Winograd removes level copies: 35.24% less user CPU and 1.750% less work](../../experiments/arithmetic-kernels/place-tree-winograd-removes-level-copies-35-24-less/README.md)
- ← shipped in: [Shared left transforms and repeated-square reuse save 0.34% work with identical predictions](../../experiments/arithmetic-kernels/shared-left-transforms-repeated-square-reuse-save-0-34/README.md)
- supersedes: [Submission #332182: upper-block covariance products, lean views, and an epoch-1 refit](../332182/README.md)
- ← supersedes: [Submission #332203: tree-based six-level reads, row padding, and general depth selection](../332203/README.md)

[Research graph](../../graph/index.html#node=submission-332195) · [Copy provenance](../../PROVENANCE.json)

