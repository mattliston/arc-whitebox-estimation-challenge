# Covariance-edge fit completes 160 graphs and 40 updates over 528 parameters with fixed mean-head bytes

2026-09-21 · experiment · informational · measured

The isolated covariance-edge fit completed its fixed training dose in separate guarded workers. It preserved all nontrainable parameters and sealed its terminal payload before validation; this execution result did not establish native accuracy.

Recorded measurement:

    160traininggraphs;40updates;528trainablevalues;40workerguards;workerpeak5995380736B;zero added swap;allworkergroups retired.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/guard.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/prepare.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/prepare.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/run_fit.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/run_fit.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/state_io.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/verify_fit.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/verify_fit.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/worker.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/worker.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/allocator.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/common.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/model_edge.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/model_edge.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/state_io.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)

</details>

## Evidence

- [fit_result.json#L1](fit_result.json#L1)

<details>
<summary>All result files and records (1)</summary>

- [fit_result.json](fit_result.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-covariance-edge-fit-completes-160-graphs-40-updates-over) · [Copy provenance](../../../PROVENANCE.json)

