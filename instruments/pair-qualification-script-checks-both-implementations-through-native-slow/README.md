# Pair-qualification script checks both implementations through native and slow-worker gates

2026-10-10T02:43Z · tool · adopted · measured

qualify_pair.sh takes an engineer file and builds BANK = file + three-request-load binding fragment (Rl3_bind128) and SAFE = file + Rl3_bind128 + fused-tree release fragment (Rlvfs_limit_off) (+ Rsym_two_strips until the lead dropped it), runs stage A on both, then the bank through stages C, D and the 1.3x slow-setup session with judge lines (about 45 min), then the safe file the same way; it stops on a quiet-window line. Stage A refuses einsum signatures new against the proven candidate generation 42 (S42) source; the three new ones of candidate generation 43 (S43) ('alsqj,lsjk,lk->aqsk' and 'a,sq,k->aqsk' in _f4c_birth, 'kn,kc->nc' in _hcf_bias) were reviewed as having no repeated operand (the symmetric-output crash law of 09-23) and passed as reviewed signatures.

Recorded measurement:

    stage A FAILS on two new einsum signatures of F4Cv2's birth, PASSES with them reviewed; one more signature for the final-layer columns; SAFE_FRAGS default 'l3_bind128 lvfs_limit_off'

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/qualify_pair.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/qualify_pair.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/variants/s42f_471a9841_lvfsoff_sym2/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/variants/s42f_471a9841_lvfsoff_sym2/candidate.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_client.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_client.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/build_variant.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/build_variant.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/native_probe.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/native_probe.py)

<details>
<summary>Additional programs and dependencies (12)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/own_gate.sh](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/own_gate.sh)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/project.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/project.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/rusage.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/rusage.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/server_light.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/server_light.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/setup_clock.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_a_package.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_a_package.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_b_native40.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_b_native40.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_c_boundary.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_c_boundary.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_d_serving.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_d_serving.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/variants/fragments/lvfs_limit_off.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/variants/fragments/lvfs_limit_off.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/variants/fragments/sym_two_strips.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/variants/fragments/sym_two_strips.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L500](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L500)
- [TEAM_LOG_2026-10-01_to_10-22.md#L505](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L505)
- [TEAM_LOG_2026-10-01_to_10-22.md#L509](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L509)
- [TEAM_LOG_2026-10-01_to_10-22.md#L510](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L510)

<details>
<summary>All result files and records (2)</summary>

- [build.json](build.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-10-pair-qualification-script-checks-both-implementations-through-native-slow) · [Copy provenance](../../PROVENANCE.json)

