# Final join shifted to layer 12 or 14: early gain reverses after expansion

2026-09-27 · experiment · rejected · measured

The final join moved at unchanged rank384 while preserving all cohorts. Layer12 passed the initial screen but failed the frozen eight-network expansion, while layer14 failed immediately.

Recorded measurement:

    Layer12 TRAIN2 adjusted-1.246599%;TRAIN8 adjusted+0.320396%,raw+2.023893%,4/8 wins;layer14 TRAIN2 adjusted+0.485368%,0/2 wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/last_join_shift_screens_v1/early12.py](code/codex_resume_20260926_v2/last_join_shift_screens_v1/early12.py)
- [code/codex_resume_20260926_v2/last_join_shift_screens_v1/late14.py](code/codex_resume_20260926_v2/last_join_shift_screens_v1/late14.py)
- [code/codex_resume_20260926_v2/last_join_shift_screens_v1/run.py](code/codex_resume_20260926_v2/last_join_shift_screens_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)
- [code/codex_resume_20260926_v2/candidate_thin_star_route_v1.py](code/codex_resume_20260926_v2/candidate_thin_star_route_v1.py)
- [code/codex_resume_20260926_v2/compare_native.py](code/codex_resume_20260926_v2/compare_native.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)

## Evidence

- [RESULTS.md#L1657](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1657)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-final-join-shifted-layer-12-or-14-early-gain) · [Copy provenance](../../../PROVENANCE.json)

