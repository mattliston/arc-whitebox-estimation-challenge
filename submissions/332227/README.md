# Submission #332227: jointly fitted physics scales with learned correction enabled

2026-09-25T03:04:13Z · submission · adopted · measured

Replaced the physics-scale literals with a joint fit made while learned correction was enabled and with consistent parameter semantics. The official public score is 2.619867019847633e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332227, "date": "2026-09-25T03:04:13Z", "status": "GRADED", "score_public": 2.619867019847633e-09, "n": 100, "failed_networks": 0, "utilization": 0.15211481314840966, "wall_mean_s": 63.062, "wall_max_s": 69.899, "resid_mean_s": 0.227, "resid_max_s": 0.248, "source_sha256": "c333be744fcc538d4cb06eccdea81e5864367744ea80bacab409eace5cddb820"}
    - 2026-09-24 20:04 PDT FIRE #332227 (Fable): fable_fire_v1/fire_candS19/candS19_S18_dials_mem.py (sha256 c333be74...) = Fable S19 = S18 #332226 + covhunter's learned-ON jointly fitted physics dials, shared semantics, literal-only:
    - 2026-09-24 20:32 PDT GRADED #332227 (S19): adjusted 2.619867019847633e-9, raw 1.7226389150692968e-8, util 0.15211. NEW PHASE-2 BANK, -1.90% vs #332226 (2.6705e-9); expected ~2.632e-9 (VALID40 -1.45%).
    - 2026-09-24 21:01 PDT GRADED #332230 (S20): adjusted 2.593723466480961e-9, raw 1.7164063397245854e-8, util 0.15115. NEW PHASE-2 BANK, -1.00% vs #332227 (2.6199e-9); expected ~2.596e-9 (VALID40 vs S19 -0.90%).

Public score: 2.619867019847633e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L809](../../records/SUBMISSION_LEDGER.md#L809)
- [SUBMISSION_LEDGER.md#L818](../../records/SUBMISSION_LEDGER.md#L818)
- [SUBMISSION_LEDGER.md#L835](../../records/SUBMISSION_LEDGER.md#L835)
- [TEAM_LOG_2026-09-22_to_09-26.md#L598](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L598)
- [TEAM_LOG_2026-09-22_to_09-26.md#L610](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L610)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332227.json](eval_332227.json) · result

</details>

## Related work

- ← shipped in: [Learned-on joint physics dials improve held-out raw error 2.23% without a refit](../../experiments/dials/learned-joint-physics-dials-improve-held-out-raw-error/README.md)
- supersedes: [Submission #332226: rank-32 memoryless (2,1) residual plane with one power iteration](../332226/README.md)
- ← supersedes: [Submission #332230: refitted physics scales and additional six-level products](../332230/README.md)

[Research graph](../../graph/index.html#node=submission-332227) · [Copy provenance](../../PROVENANCE.json)

