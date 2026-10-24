# Graph-feature linear augmentation increases cross-validated raw error 0.066533%

2026-09-30 · experiment · rejected · measured

A fixed57-feature graph augmentation was tested with linear ridge cross-validation. It failed all prespecified scientific gates, so no full-data fit, export or native port followed.

Recorded measurement:

    Raw+.066533%,adjusted proxy+.062160%,1/5fold wins;guard1.0303s,RSS208683008B,zero swap.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_rrpi_graph57_ridge_v1/fit.py](code/codex_resume_20260926_v2/hill_climb_rrpi_graph57_ridge_v1/fit.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/disabled_control.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/disabled_control.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/fit.py](code/codex_resume_20260926_v2/hill_climb_rrpi_head160_cached_v1/fit.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/graph.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/graph.py)
- [code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/source_edit.py](code/codex_resume_20260926_v2/teammate_terminal_head_adapt_v1/source_edit.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L217](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L217)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [protocol.json](protocol.json) · result
- [cv_result.json](cv_result.json) · result
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-graph-feature-linear-augmentation-increases-cross-validated-raw-error) · [Copy provenance](../../../PROVENANCE.json)

