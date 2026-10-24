# Native final-error graph head: fitted gain reverses to 2.97073% raw loss on 32 check networks

2026-09-27 · experiment · rejected · measured

A nonlinear neighbor/subspace final-error head was fitted on128 native captures and checked on32 reused training networks. Both the graph and local-feature heads failed fixed information and paid-screen gates.

Recorded measurement:

    Graph FIT raw-4.84726%,CHECK+2.97073%,95% interval[+2.31249%,+3.67570%],2/32 wins;control FIT-0.50001%,CHECK+0.07575%,interval[-0.16869%,+0.33286%],16/32 wins;graph/control+2.89280%;40 epochs.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/native_final_graph_head_v1/assess.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/assess.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/build_native.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/build_native.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/capture.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/capture.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/finish.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/finish.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/fit.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/fit.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/graph.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/graph.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/model.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/model.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/native_helper_draft.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/native_helper_draft.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/qualify.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/qualify.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/queue_capture.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/queue_capture.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/record_checkpoint24.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/record_checkpoint24.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/select_survivor.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/select_survivor.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/codex_resume_20260926_v2/native_final_graph_head_v1/select_survivor_before_banking_v2.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/select_survivor_before_banking_v2.py)
- [code/codex_resume_20260926_v2/native_final_graph_head_v1/status.py](code/codex_resume_20260926_v2/native_final_graph_head_v1/status.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)
- [code/codex_resume_20260926_v2/candidate_thin_star_route_v1.py](code/codex_resume_20260926_v2/candidate_thin_star_route_v1.py)

</details>

## Evidence

- [RESULTS.md#L1153](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1153)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-native-final-error-graph-head-fitted-gain-reverses-2) · [Copy provenance](../../../PROVENANCE.json)

