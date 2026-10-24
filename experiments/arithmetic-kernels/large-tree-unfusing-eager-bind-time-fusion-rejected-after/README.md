# Large-tree unfusing and eager bind-time fusion rejected after request and FLOP measurements

2026-10-10T02:29Z · experiment · rejected · measured

Besides the adopted no-un-fused-trees variant (a), two other request-diet forms of the candidate generation 42 (S42) file were priced under the platform's setup clock: (b) un-fusing only trees above 64M floats (Rlvfs_limit64) saves -498 / -520 requests (first / warm) at metered FLOPs +0.004% / +0.02%, and (c) fusing keys 129-300 as they are bound (_PB_EAGER 300) is worse by +134 / +113 requests and +0.04% metered FLOPs because fusion at bind time slows the binding so 47 more keys fall to lazy binds. (b) was superseded by (a); (c) was closed.

Recorded measurement:

    fired S42 43,696 / 42,743 requests; (b) 43,198 / 42,223 = -498 / -520, bill +0.004% / +0.02%, wall 79.5 / 85.8 s; (c) +134 / +113 requests, bill +0.04%, 47 more keys to lazy binds

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/release/variant_table.py](code/opus_reexam_20261001_v1/release/variant_table.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/release/boundary_client.py](code/opus_reexam_20261001_v1/release/boundary_client.py)
- [code/opus_reexam_20261001_v1/release/boundary_run.py](code/opus_reexam_20261001_v1/release/boundary_run.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/project.py](code/opus_reexam_20261001_v1/release/project.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/server_light.py](code/opus_reexam_20261001_v1/release/server_light.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s41hv_33bac9db_lvfs64_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s41hv_33bac9db_lvfs64_l3b128/candidate.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L493](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L493)

<details>
<summary>All result files and records (11)</summary>

- [busy_at_start.json](busy_at_start.json) · result
- [guard.json](guard.json) · result
- [guard.log](guard.log) · log
- [launcher.json](launcher.json) · result
- [manifest.json](manifest.json) · result
- [rows.jsonl](rows.jsonl) · result
- [server.json](server.json) · result
- [server.log](server.log) · log
- [server_marks.jsonl](server_marks.jsonl) · result
- [stage_c.json](stage_c.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-large-tree-unfusing-eager-bind-time-fusion-rejected-after) · [Copy provenance](../../../PROVENANCE.json)

