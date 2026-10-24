# Layer-14 Gaussian fourth-order restoration completed without a recorded final assessment

2026-10-01 · experiment · inconclusive · measured

The final full pair layer gains the fixed rho⁴/24 Gaussian term in the covariance, square-linear and square-square raw channels before the existing centralization and diagonal resets. Although the last checkpoint says the run is live, the saved guard and four per-network result rows establish clean completion. The source snapshot contains no final assessor output, broader check or retention decision; only those recorded individual measurements are reported.

Recorded measurement:

    Recorded native rows: rank 35: final-layer MSE 1.4597391025225421e-08, utilization 0.11959034231040278, adjusted 1.7457069895455093e-09, billed operations 262981943880; rank 100: final-layer MSE 1.527281495885615e-08, utilization 0.1198933143823524, adjusted 1.8311084053656348e-09, billed operations 263648186512; rank 36: final-layer MSE 1.682108466742651e-08, utilization 0.12069554227628032, adjusted 2.0302299356102667e-09, billed operations 265412304307; rank 45: final-layer MSE 1.4963786878846852e-08, utilization 0.1181184145334555, adjusted 1.7674987815459148e-09, billed operations 259745140468. Recorded guard: completed=True, exit_code=0, elapsed_s=103.5841022497043, max_rss_bytes=5689688064, max_swap_growth_bytes=0, whole_group_retired=True. No recorded aggregate assessment.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_gaussian4_l14_v1/candidate.py](code/hill_climb_rrpi_gaussian4_l14_v1/candidate.py)
- [code/hill_climb_rrpi_gaussian4_l14_v1/candidate_identity.py](code/hill_climb_rrpi_gaussian4_l14_v1/candidate_identity.py)
- [code/hill_climb_rrpi_gaussian4_l14_v1/build.py](code/hill_climb_rrpi_gaussian4_l14_v1/build.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L2](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L2)
- [HILL_CLIMB_CURRENT.md#L5](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L5)
- [HILL_CLIMB_CURRENT.md#L6](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L6)
- [identity_assessment.json#L35](identity_assessment.json#L35)
- [rows.jsonl#L1](rows.jsonl#L1)
- [rows.jsonl#L2](rows.jsonl#L2)
- [rows.jsonl#L3](rows.jsonl#L3)
- [rows.jsonl#L4](rows.jsonl#L4)
- [rows-ee49e16a.jsonl#L1](rows-ee49e16a.jsonl#L1)
- [rows-ee49e16a.jsonl#L2](rows-ee49e16a.jsonl#L2)
- [rows-ee49e16a.jsonl#L3](rows-ee49e16a.jsonl#L3)
- [rows-ee49e16a.jsonl#L4](rows-ee49e16a.jsonl#L4)

<details>
<summary>All result files and records (9)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [source_delta.json](source_delta.json) · result
- [source_review.json](source_review.json) · result
- [identity_assessment.json](identity_assessment.json) · result
- [native_fit4_plan.json](native_fit4_plan.json) · record
- [native_fit4_guard.json](native_fit4_guard.json) · result
- [native_fit4.log](native_fit4.log) · log
- [rows.jsonl](rows.jsonl) · result
- [rows-ee49e16a.jsonl](rows-ee49e16a.jsonl) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-layer-14-gaussian-fourth-order-restoration-completed-without-recorded) · [Copy provenance](../../../PROVENANCE.json)

