# Native builder exports learned head rows and maps covariance coefficient kc = 1 - kappa

2026-10-10T06:18Z · tool · adopted · measured

stack/s43_build.sh <hand-over> <head table> <base> turns a trainer hand-over (candidate.py with the refitted learned arrays, physics_dials.json, a head-weight table in the HEAD_COLUMNS row order) into the native file: weights_from_trainer.py maps rows 6, 9, 12-16, 20, 21, 27, 28, 29 to the layer 1-14 HC lever and rows 17-19 to the layer-15 HCF lever (rows 22-26 and 30 have no native form and are refused), mean-correction covariance-fold coefficient (kappa_C) per layer becomes the native kc_l = 1 - kappa_l on top of the moment-closure estimator's own rc = -0.08 (C = C0 - (1 + rc_l - kc_l) S), and band rows 31-35 go to the BAND lever. Every candidate generation 43 (S43)-candidate generation 46 (S46) build (candidate generation 44 (S44) sha 67dfcf71, jmD 19251d3a, jmG 85d0c9af, joint global-dial and learned-array refit (jmH) b36b7154) came from it, followed by per-layer parity against the trainer's PyTorch predictions on networks 651 / 378.

Recorded measurement:

    converter unit check exact on a synthetic table; kappa 1 - 0.1 (l mod 3) -> kc 0.1 (l mod 3); file name 240 bytes < 244; S44 build sha 67dfcf71 from jmC head_weights_31x128_without_qold (row norms 0.389 / 0.555 / 0.197 / 0.018 / 0.305)

## Programs

- [code/opus_reexam_20261001_v1/engineer/stack/s43_build.sh](code/opus_reexam_20261001_v1/engineer/stack/s43_build.sh)
- [code/opus_reexam_20261001_v1/engineer/stack/law_file_from_trainer.sh](code/opus_reexam_20261001_v1/engineer/stack/law_file_from_trainer.sh)
- [code/opus_reexam_20261001_v1/engineer/a42_head/weights_from_trainer.py](code/opus_reexam_20261001_v1/engineer/a42_head/weights_from_trainer.py)
- [code/opus_reexam_20261001_v1/engineer/a47_kc/kc_from_trainer.py](code/opus_reexam_20261001_v1/engineer/a47_kc/kc_from_trainer.py)
- [code/opus_reexam_20261001_v1/engineer/a47_kc/fragment_kc_v1.py](code/opus_reexam_20261001_v1/engineer/a47_kc/fragment_kc_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a49_band/fragment_band_v1.py](code/opus_reexam_20261001_v1/engineer/a49_band/fragment_band_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/fable_porter_v1/einsum_site_diff.py](code/fable_porter_v1/einsum_site_diff.py)
- [code/opus_reexam_20261001_v1/engineer/a10_bd32/fragment_bd32.py](code/opus_reexam_20261001_v1/engineer/a10_bd32/fragment_bd32.py)
- [code/opus_reexam_20261001_v1/engineer/a11_bare_views/analyze_binds.py](code/opus_reexam_20261001_v1/engineer/a11_bare_views/analyze_binds.py)
- [code/opus_reexam_20261001_v1/engineer/a11_bare_views/fragment_bvp.py](code/opus_reexam_20261001_v1/engineer/a11_bare_views/fragment_bvp.py)
- [code/opus_reexam_20261001_v1/engineer/a12_live_slice/fragment_dcr.py](code/opus_reexam_20261001_v1/engineer/a12_live_slice/fragment_dcr.py)

<details>
<summary>Additional programs and dependencies (45)</summary>

- [code/opus_reexam_20261001_v1/engineer/a17_tanh/fragment_ets.py](code/opus_reexam_20261001_v1/engineer/a17_tanh/fragment_ets.py)
- [code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut.py](code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut.py)
- [code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut_v2.py](code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3.py](code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3.py)
- [code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3o.py](code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3o.py)
- [code/opus_reexam_20261001_v1/engineer/a29_e2/fragment_e2_v1.py](code/opus_reexam_20261001_v1/engineer/a29_e2/fragment_e2_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a31_star/fragment_sx_v1.py](code/opus_reexam_20261001_v1/engineer/a31_star/fragment_sx_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a32_c2k/fragment_c2k_v1.py](code/opus_reexam_20261001_v1/engineer/a32_c2k/fragment_c2k_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/fragment_ss_v3.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/fragment_ss_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a35_join/fragment_jr_v1.py](code/opus_reexam_20261001_v1/engineer/a35_join/fragment_jr_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a38_pairlaw/dials_from_trainer.py](code/opus_reexam_20261001_v1/engineer/a38_pairlaw/dials_from_trainer.py)
- [code/opus_reexam_20261001_v1/engineer/a40_setup/fragment_cs_v1.py](code/opus_reexam_20261001_v1/engineer/a40_setup/fragment_cs_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bk3_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bk3_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bsk_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_bsk_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_gsk_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_gsk_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v1.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v2.py](code/opus_reexam_20261001_v1/engineer/a41_base/fragment_so2_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v1.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v2.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hc_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v1.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v3.py](code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a43_readout/fragment_ro_v1.py](code/opus_reexam_20261001_v1/engineer/a43_readout/fragment_ro_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v1.py](code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v2.py](code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v3.py](code/opus_reexam_20261001_v1/engineer/a44_cov/fragment_dcv_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a45_l15/fragment_l15_v1.py](code/opus_reexam_20261001_v1/engineer/a45_l15/fragment_l15_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a46_b6/fragment_b6_v1.py](code/opus_reexam_20261001_v1/engineer/a46_b6/fragment_b6_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a46_b6/fragment_b6_v2.py](code/opus_reexam_20261001_v1/engineer/a46_b6/fragment_b6_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a50_b7/fragment_b7_v2.py](code/opus_reexam_20261001_v1/engineer/a50_b7/fragment_b7_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a52_pld/dly_from_trainer.py](code/opus_reexam_20261001_v1/engineer/a52_pld/dly_from_trainer.py)
- [code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v2.py](code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v3.py](code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a55_u15/fragment_u15_v1.py](code/opus_reexam_20261001_v1/engineer/a55_u15/fragment_u15_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a56_req/fragment_rqd_v1.py](code/opus_reexam_20261001_v1/engineer/a56_req/fragment_rqd_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a56_req/fragment_rqd_v2.py](code/opus_reexam_20261001_v1/engineer/a56_req/fragment_rqd_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a56_req/fragment_rqs_v1.py](code/opus_reexam_20261001_v1/engineer/a56_req/fragment_rqs_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a57_hz7/fragment_hz7_v1.py](code/opus_reexam_20261001_v1/engineer/a57_hz7/fragment_hz7_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a57_hz7/fragment_hz7_v3.py](code/opus_reexam_20261001_v1/engineer/a57_hz7/fragment_hz7_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a57_hz7/fragment_hz7_v4.py](code/opus_reexam_20261001_v1/engineer/a57_hz7/fragment_hz7_v4.py)
- [code/opus_reexam_20261001_v1/engineer/a60_psdm/fragment_psdm_v1.py](code/opus_reexam_20261001_v1/engineer/a60_psdm/fragment_psdm_v1.py)
- [code/opus_reexam_20261001_v1/engineer/buildlib.py](code/opus_reexam_20261001_v1/engineer/buildlib.py)
- [code/opus_reexam_20261001_v1/engineer/edits_lib.py](code/opus_reexam_20261001_v1/engineer/edits_lib.py)
- [code/opus_reexam_20261001_v1/engineer/hzs/fragment_gen_v7.py](code/opus_reexam_20261001_v1/engineer/hzs/fragment_gen_v7.py)
- [code/opus_reexam_20261001_v1/engineer/stack/build.py](code/opus_reexam_20261001_v1/engineer/stack/build.py)
- [code/opus_reexam_20261001_v1/engineer/stack/candidate_T_C_B_F_D_Hfull_F4Ct0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv4c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_NPf3bv4_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_cand3_five.py](code/opus_reexam_20261001_v1/engineer/stack/candidate_T_C_B_F_D_Hfull_F4Ct0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv4c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_NPf3bv4_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_cand3_five.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L541](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L541)
- [TEAM_LOG_2026-10-01_to_10-22.md#L545](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L545)
- [TEAM_LOG_2026-10-01_to_10-22.md#L567](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L567)
- [TEAM_LOG_2026-10-01_to_10-22.md#L575](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L575)
- [TEAM_LOG_2026-10-01_to_10-22.md#L593](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L593)

<details>
<summary>All result files and records (2)</summary>

- [head_weights_31x128_without_qold.json](head_weights_31x128_without_qold.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-10-native-builder-exports-learned-head-rows-maps-covariance-coefficient) · [Copy provenance](../../PROVENANCE.json)

