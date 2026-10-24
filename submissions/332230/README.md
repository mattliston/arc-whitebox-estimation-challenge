# Submission #332230: refitted physics scales and additional six-level products

2026-09-25T03:34:26Z · submission · adopted · measured

Added another consistent-semantics physics-scale fit and moved remaining eligible products to six arithmetic levels. The official public score is 2.593723466480961e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332230, "date": "2026-09-25T03:34:26Z", "status": "GRADED", "score_public": 2.593723466480961e-09, "n": 100, "failed_networks": 0, "utilization": 0.15114723904313906, "wall_mean_s": 65.433, "wall_max_s": 70.486, "resid_mean_s": 0.233, "resid_max_s": 0.251, "source_sha256": "4682124074bd4e93bb77f33306965fd65234444c3282feb74397623e618cd54f"}
    - 2026-09-24 20:34 PDT FIRE #332230 (Fable): fable_fire_v1/fire_candS20/candS20_S18_dials2_d6x_mem.py (sha256 46821240...) = Fable S20 = S18 #332226 + covhunter's shared-semantics refit dials (c0 .653, c1 1.608, k4 .988/.962/.963, j1 .533, j2 .213, j3 -.228, b 1.049, lam 1.0044/.9932) + engineer2's d6x (remaining depth-6 moves).
    - 2026-09-24 21:01 PDT GRADED #332230 (S20): adjusted 2.593723466480961e-9, raw 1.7164063397245854e-8, util 0.15115. NEW PHASE-2 BANK, -1.00% vs #332227 (2.6199e-9); expected ~2.596e-9 (VALID40 vs S19 -0.90%).
    - 2026-09-24 21:21 PDT GRADED #332231 (S21): adjusted 2.5832095956918383e-9, raw 1.7088557644484582e-8, util 0.15119. NEW PHASE-2 BANK, -0.40% vs #332230 (2.5937e-9), -1.40% vs S19; expected ~2.58e-9 (VALID40 vs S19 -1.46%).

Public score: 2.593723466480961e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L822](../../records/SUBMISSION_LEDGER.md#L822)
- [SUBMISSION_LEDGER.md#L835](../../records/SUBMISSION_LEDGER.md#L835)
- [SUBMISSION_LEDGER.md#L839](../../records/SUBMISSION_LEDGER.md#L839)
- [TEAM_LOG_2026-09-22_to_09-26.md#L615](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L615)
- [TEAM_LOG_2026-09-22_to_09-26.md#L622](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L622)
- [TEAM_LOG_2026-09-22_to_09-26.md#L625](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L625)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332230.json](eval_332230.json) · result

</details>

## Related work

- ← shipped in: [Additional eligible depth-six products save 0.54% work for 3.95% more user CPU](../../experiments/arithmetic-kernels/additional-eligible-depth-six-products-save-0-54-work/README.md)
- ← shipped in: [Consistent-semantics physics refit improves raw error 1.98% on forty low-rank residual-plane networks](../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/README.md)
- supersedes: [Submission #332227: jointly fitted physics scales with learned correction enabled](../332227/README.md)
- ← supersedes: [Submission #332231: epoch-2 learned parameters with first-round physics scales](../332231/README.md)

[Research graph](../../graph/index.html#node=submission-332230) · [Copy provenance](../../PROVENANCE.json)

