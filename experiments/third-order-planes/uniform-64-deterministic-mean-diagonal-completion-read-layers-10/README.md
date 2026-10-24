# Uniform-64 deterministic mean and diagonal completion of the read at layers 10-14: raw +1.09% for bill -0.40%, closed

2026-10-17T09:33:20Z · experiment · rejected · measured

Reducing the read rank by 64 at layers 10-14 while preserving the exact missing-row mean before the diagonal projection recovered 26.9% of the saved leading symmetric covariance error offline, but the one authorized native pilot on generated network 35 lost accuracy faster than it saved bill and added requests and residual instructions; closed without tuning.

Recorded measurement:

    Saved 651 recovery 26.9466%. Pilot 35 vs matched current-SDK control: raw +1.089470%, bill -.404613% (-986.23 MF), adjusted +.680448%, +451 requests, +1.42075% residual instructions.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/uniform64_diag_v1/run_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/uniform64_diag_v1/run_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/uniform64_diag_v1/candidate_v25_uniform64_md_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/uniform64_diag_v1/candidate_v25_uniform64_md_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/uniform64_diag_v1/client_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/uniform64_diag_v1/client_v1.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4863](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4863)

<details>
<summary>All result files and records (3)</summary>

- [PILOT35_RESULT_v1.json](PILOT35_RESULT_v1.json) · result
- [CLOSURE_v1.json](CLOSURE_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-uniform-64-deterministic-mean-diagonal-completion-read-layers-10) · [Copy provenance](../../../PROVENANCE.json)

