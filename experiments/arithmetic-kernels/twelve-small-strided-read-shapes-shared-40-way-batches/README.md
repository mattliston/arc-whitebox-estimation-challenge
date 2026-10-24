# Twelve small strided-read shapes on shared 40-way batches: eight-network raw -0.199%; expansions lose

2026-10-17T01:11:19Z · experiment · adopted · measured

Twelve fixed small Smirnov read shapes routed through two shared layer-5 P40 families (0.1626702611% warm metadata saving, -211.875 API calls per network, setup 859 RPC) passed the two authorized boundary pilots and then a fixed 8 expansion with a consistent raw gain; its bill credit was later narrowed to 0.108784-0.113185% from the actual accepted shapes. Two expansions were closed as adverse: 28 cost-selected old-inner shapes at layer 4 (+0.1775% mean raw) and 24 shapes reusing the existing layer-5 frames (+0.2455% mean adjusted).

Recorded measurement:

    Boundary 35+150 raw U vs full 48c: -0.344596% / -0.166079%; fixed 8 incremental raw mean -0.1985049% SE 0.0707985%, pooled -0.1838610%, 7/8 wins; projected setups 4.271 s / 19,100 RPC. P40 recovery 0.108784-0.113185%. L4 additional 28 (02e76662): raw +0.127722% / +0.227290%, mean +0.177506%. L5 additional 24 (240d3b09): raw +0.302826% / +0.460417%, mean adjusted +0.245524%. HK15 residual after peel overlap: max remaining tariff .031826%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_residual_v1/boundary_run_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_residual_v1/boundary_run_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_residual_v1/candidate_s2tr_peel_sm12_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_residual_v1/candidate_s2tr_peel_sm12_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/boundary_run_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/boundary_run_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/candidate_s2tr_peel_sm28_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/candidate_s2tr_peel_sm28_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/boundary_run_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/boundary_run_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/candidate_s2tr_peel_sm24_L5_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/candidate_s2tr_peel_sm24_L5_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_reexam_20261001_v1/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/boundary_client_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/boundary_client_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/boundary_client_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_expand_v1/boundary_client_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_residual_v1/boundary_client_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_smallsm_residual_v1/boundary_client_v1.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4604](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4604)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4606](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4606)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4607](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4607)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4608](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4608)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4609](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4609)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4610](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4610)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4611](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4611)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4643](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4643)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4647](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4647)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4697](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4697)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4699](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4699)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4701](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4701)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4704](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4704)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4706](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4706)

<details>
<summary>All result files and records (14)</summary>

- [PILOT_RESULT_v1.json](PILOT_RESULT_v1.json) · result
- [FIXED8_RESULT_v1.json](../../../records/research/2026-10-17-depth-peeling-exact-cubic-messages-four-coefficients-sixteen-network/FIXED8_RESULT_v1.json) · result
- [P40_RECOVERY_v1.json](../../../records/research/2026-10-17-depth-peeling-exact-cubic-messages-four-coefficients-sixteen-network/P40_RECOVERY_v1.json) · result
- [PILOT_RESULT_v1-c4882583.json](PILOT_RESULT_v1-c4882583.json) · result
- [PILOT_L5_RESULT_v1.json](PILOT_L5_RESULT_v1.json) · result
- [AUDIT_v1.json](AUDIT_v1.json) · record
- [BOUNDARY_GUARD_v1.json](BOUNDARY_GUARD_v1.json) · result
- [guard_16_fixed8_v1.json](guard_16_fixed8_v1.json) · result
- [guard_200_fixed8_v1.json](guard_200_fixed8_v1.json) · result
- [guard_250_fixed8_v1.json](guard_250_fixed8_v1.json) · result
- [guard_259_fixed8_v1.json](guard_259_fixed8_v1.json) · result
- [guard_277_fixed8_v1.json](guard_277_fixed8_v1.json) · result
- [guard_287_fixed8_v1.json](guard_287_fixed8_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-twelve-small-strided-read-shapes-shared-40-way-batches) · [Copy provenance](../../../PROVENANCE.json)

