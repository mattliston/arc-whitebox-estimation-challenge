# Deployed-edge off-diagonal mean projection: adjusted error rises 0.334747%

2026-09-27 · experiment · rejected · measured

The deployed edge correction was projected off the mean in its off-diagonal channel. Independent exact/source checks passed, but the actual native scores did not improve.

Recorded measurement:

    TRAIN2 raw+0.918150%,adjusted+0.334747%,0wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/teammate_edge_projection_20260927_v1/build.py](code/teammate_edge_projection_20260927_v1/build.py)
- [code/teammate_edge_projection_20260927_v1/candidate.py](code/teammate_edge_projection_20260927_v1/candidate.py)
- [code/teammate_edge_projection_20260927_v1/check_guard.py](code/teammate_edge_projection_20260927_v1/check_guard.py)
- [code/teammate_edge_projection_20260927_v1/fragment.py](code/teammate_edge_projection_20260927_v1/fragment.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/candidate_thin_star_route_v1.py](code/candidate_thin_star_route_v1.py)
- [code/current_covariance_pair_supervision_v1/projection.py](code/current_covariance_pair_supervision_v1/projection.py)
- [code/native_covariance_innovation_fit_v1/candidate.py](code/native_covariance_innovation_fit_v1/candidate.py)

## Evidence

- [NEXT_RESEARCH_v1.md#L19](../../dials/covariance-convention-correction-strength-three-short-screens-stay-below/NEXT_RESEARCH_v1.md#L19)

<details>
<summary>All result files and records (1)</summary>

- [NEXT_RESEARCH_v1.md](../../dials/covariance-convention-correction-strength-three-short-screens-stay-below/NEXT_RESEARCH_v1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-deployed-edge-off-diagonal-mean-projection-adjusted-error-rises) · [Copy provenance](../../../PROVENANCE.json)

