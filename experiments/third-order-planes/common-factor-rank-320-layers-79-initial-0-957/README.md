# Common-factor rank 320 at layers 7–9: initial 0.957% gain shrinks to 0.134968%

2026-09-28 · experiment · rejected · measured

A fixed16-network follow-up rejected the rank320 early-layer extension after the initial two-network gain failed to persist.

Recorded measurement:

    TRAIN2 gain0.957%;TRAIN16 increment-0.134968%,9wins,CI[-0.358910%,+0.080031%].

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_common320_l7_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_common320_l7_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_common320_l7_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_common320_l7_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_common320_l7_v1/run_followup.py](code/codex_resume_20260926_v2/hill_climb_common320_l7_v1/run_followup.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/compare_native.py](code/codex_resume_20260926_v2/compare_native.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)

## Evidence

- [HILL_CLIMB_CURRENT_before_332738.md#L36](../common-factor-rank-256-layers-79-16-network-adjusted/HILL_CLIMB_CURRENT_before_332738.md#L36)

<details>
<summary>All result files and records (1)</summary>

- [HILL_CLIMB_CURRENT_before_332738.md](../common-factor-rank-256-layers-79-16-network-adjusted/HILL_CLIMB_CURRENT_before_332738.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-common-factor-rank-320-layers-79-initial-0-957) · [Copy provenance](../../../PROVENANCE.json)

