# Birth-direction arithmetic narrowed to float32: fixed16 adjusted gain is only 0.054926441%

2026-09-28 · experiment · rejected · measured

Birth-direction arithmetic was ported to float32 while preserving the float64 physical-delta formula. The initial gain disappeared on expansion.

Recorded measurement:

    TRAIN2 adjusted-.4433035%,2wins;TRAIN16-.054926441%,raw+.084705249%,bill-.137098799%,10wins,CI[-.304731155,+.201819389],additional14-.006804738%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_birth_direction_fp32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_birth_direction_fp32_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_birth_direction_fp32_v1/build.py](code/codex_resume_20260926_v2/hill_climb_birth_direction_fp32_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_birth_native_assess_v1.py](code/codex_resume_20260926_v2/hill_climb_birth_native_assess_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
- [code/fable_engineer_v1/build_f32.py](code/fable_engineer_v1/build_f32.py)

## Evidence

- [HILL_CLIMB_before_tree_birth13_v1.md#L22](../birth-directions-selected-distinct-third-moment-strength-fixed16-gain/HILL_CLIMB_before_tree_birth13_v1.md#L22)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_before_tree_birth13_v1.md](../birth-directions-selected-distinct-third-moment-strength-fixed16-gain/HILL_CLIMB_before_tree_birth13_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [train16_assessment.json](train16_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-birth-direction-arithmetic-narrowed-float32-fixed16-adjusted-gain-only) · [Copy provenance](../../../PROVENANCE.json)

