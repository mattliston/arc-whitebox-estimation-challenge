# Two exact setup offloads: terminal Smirnov scale batch (-8/-9 RPC per circuit) and 64 main-read aliases in setup

2026-10-17T02:20:12Z · experiment · adopted · measured

Two byte-exact components were proved on the actual client for the runtime ladder: the terminal scale groups of the Smirnov P circuits replaced by one public multiply of the unscaled outputs with fixed f32 coefficients (-8 or -9 requests per circuit, fewer FLOPs, 0.01% conditional credit on the science 16), and 64 fixed exact main-read V aliases built in setup (128 RPC and 825.477120 MF off-score, cold 35 -11 RPC, post-first -55 RPC). Both were integrated in the v16/v17 setup.

Recorded measurement:

    Terminal scale: e1 -8 RPC / -9 mn F, e0 -9 RPC / -18 mn F; 35/100 routes save 90/124 RPC and 23.151/28.404 MF; science 16 credit .01002065% (.01069407% proxy-inclusive). Main-read reshape-only setup: 128 RPC / 825.477120 MF to setup; cold 35 -11 RPC / 151.265280 MF; post-first -55 RPC / 696.002560 MF (1,818 -> 1,763); warm 100 unchanged.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/terminal_scale_v1/run_tiny_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/terminal_scale_v1/run_tiny_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/mainread_reshape_setup_v1/run_tiny_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/mainread_reshape_setup_v1/run_tiny_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/mainread_reshape_setup_v1/client_tiny_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/mainread_reshape_setup_v1/client_tiny_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/terminal_scale_v1/client_tiny_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/terminal_scale_v1/client_tiny_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v13.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v13.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4734](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4734)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4736](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4736)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4746](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4746)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4748](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4748)

<details>
<summary>All result files and records (3)</summary>

- [GUARD_v1.json](GUARD_v1.json) · code
- [GUARD_v2.json](GUARD_v2.json) · code
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-two-exact-setup-offloads-terminal-smirnov-scale-batch-8) · [Copy provenance](../../../PROVENANCE.json)

