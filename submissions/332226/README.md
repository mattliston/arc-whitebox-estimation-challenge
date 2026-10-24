# Submission #332226: rank-32 memoryless (2,1) residual plane with one power iteration

2026-09-25T02:01:44Z · submission · adopted · measured

Replaced the dense memoryless (2,1) residual plane with a rank-32 representation using 48 sampled columns and one power iteration. The official public score is 2.670488110920038e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332226, "date": "2026-09-25T02:01:44Z", "status": "GRADED", "score_public": 2.670488110920038e-09, "n": 100, "failed_networks": 0, "utilization": 0.1519868374933958, "wall_mean_s": 64.545, "wall_max_s": 67.93, "resid_mean_s": 0.234, "resid_max_s": 0.247, "source_sha256": "ab8c8ba8665344918ba70b4eeb955fe2a9f234479e44240facd7e6dc93529fa4"}
    - 2026-09-24 19:32 PDT FIRE #332226 (Fable): fable_fire_v1/fire_candS18/candS18_S17_lrq1r32_mem.py (sha256 ab8c8ba8...) = Fable S18 = S17 #332224 + low-rank star plane.
    - CORRECTION (lead, 19:18 PDT): the #332226 (S18) FIRE entry is stamped 19:32; the actual submission time was 19:01 PDT (submit_output.txt mtime / first poll line). S16 16:12 and S17 18:34 are correct to about 2 min.
    - 2026-09-24 ~19:25 PDT GRADED #332226 (S18): adjusted 2.670488110920038e-9, raw 1.7577477251506934e-8, util 0.15199. NEW PHASE-2 BANK, -5.24% vs #332224 (2.8182e-9); expected ~2.667e-9.
    - 2026-09-24 20:04 PDT FIRE #332227 (Fable): fable_fire_v1/fire_candS19/candS19_S18_dials_mem.py (sha256 c333be74...) = Fable S19 = S18 #332226 + covhunter's learned-ON jointly fitted physics dials, shared semantics, literal-only:
    - 2026-09-24 20:32 PDT GRADED #332227 (S19): adjusted 2.619867019847633e-9, raw 1.7226389150692968e-8, util 0.15211. NEW PHASE-2 BANK, -1.90% vs #332226 (2.6705e-9); expected ~2.632e-9 (VALID40 -1.45%).
    - 2026-09-24 20:34 PDT FIRE #332230 (Fable): fable_fire_v1/fire_candS20/candS20_S18_dials2_d6x_mem.py (sha256 46821240...) = Fable S20 = S18 #332226 + covhunter's shared-semantics refit dials (c0 .653, c1 1.608, k4 .988/.962/.963, j1 .533, j2 .213, j3 -.228, b 1.049, lam 1.0044/.9932) + engineer2's d6x (remaining depth-6 moves).
    - 2026-09-24 20:56 PDT FIRE #332231 (Fable): fable_fire_v1/fire_candS21/candS21_S18ep2_dials_d6x_mem.py (sha256 eddbcfcd...) = Fable S21 = S18 #332226 + physicist's j05 ep2 refit literals (28 literals bit-equal to params) + covhunter's round-1 shared-semantics dials (c0 .709, c1 .503, k4 .989/.963/.960, X1 .559, X2 .358, X3 .152, b 1.051, lam 1.0047/.9931) + engineer2's d6x. Built by the porter's chain.

Public score: 2.670488110920038e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L797](../../records/SUBMISSION_LEDGER.md#L797)
- [SUBMISSION_LEDGER.md#L804](../../records/SUBMISSION_LEDGER.md#L804)
- [SUBMISSION_LEDGER.md#L805](../../records/SUBMISSION_LEDGER.md#L805)
- [SUBMISSION_LEDGER.md#L809](../../records/SUBMISSION_LEDGER.md#L809)
- [SUBMISSION_LEDGER.md#L818](../../records/SUBMISSION_LEDGER.md#L818)
- [SUBMISSION_LEDGER.md#L822](../../records/SUBMISSION_LEDGER.md#L822)
- [SUBMISSION_LEDGER.md#L829](../../records/SUBMISSION_LEDGER.md#L829)
- [TEAM_LOG_2026-09-22_to_09-26.md#L570](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L570)
- [TEAM_LOG_2026-09-22_to_09-26.md#L584](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L584)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332226.json](eval_332226.json) · result

</details>

## Related work

- ← shipped in: [Rank-32 residual-plane transport with one power iteration improves adjusted error 5.352%](../../experiments/third-order-planes/rank-32-residual-plane-transport-one-power-iteration-improves/README.md)
- ← shipped in: [Residual-plane transport cannot be skipped, but exact rank-32 truncation costs only 0.375% raw error](../../experiments/third-order-planes/residual-plane-transport-cannot-be-skipped-but-exact-rank/README.md)
- supersedes: [Submission #332224: half-strength unit birth feed, refit, lean views, and setup prebinding](../332224/README.md)
- ← supersedes: [Submission #332227: jointly fitted physics scales with learned correction enabled](../332227/README.md)

[Research graph](../../graph/index.html#node=submission-332226) · [Copy provenance](../../PROVENANCE.json)

