# Per-layer carried-column scale hook: byte-identical at identity, parity cosine 0.98-0.9998

2026-10-11T09:14Z · tool · adopted · measured

For the trainer's eleventh public-gradient refit after Nd (Ne) arm (s_carry -0.0039 at layers 1-6) the engineer built a per-layer s_carry hook (DLY values key 'scarry', fragment_pld_v3.py) entering four native sites: the loop's carried lam at the end of layer l, the F4C step's lam2, the pair law's per-layer constants aV / aG / iaV / iaG (rebuilt once the per-layer dict exists) and the F4C birth mix divided by the dials a cohort collects (s_carry ** (age+1) becomes the product of layers l+1 .. l+age+1). Identity at 32-network ridge-re-sized cut-simulated refit (Nd)'s value is byte-identical to the candidate generation 48 (S48) base; the +0.01 test arm at layers 1-6 moved the final-layer MSE +1.31% / +0.62% on 651 / 378 and matched the trainer's cut-simulated PyTorch change with cosine 0.9997 / 0.9998 at layer 2 and 0.984-0.994 at layers 3-15 (size ratio 0.999-1.031); the Ne trial arm matched at cosine 0.998 (layer 2) and 0.92-0.96 (layers 6-15). s48_build.sh reads pl['s_carry'] from a checkpoint through dly_from_pt_v2.py. The convention: s_carry(l) scales the columns carried out of layer l; the cohort born at l takes s_new(l).

Recorded measurement:

    identity byte-identical (predictions, bill, ops) on 651 / 378; +0.01 arm MSE +1.31% / +0.62%, bill unchanged, ops +12 / 0; parity cosine 0.9997 / 0.9998 (L2), 0.984-0.994 (L3-15), size ratio 0.999-1.031; Ne arm cosine 0.998 (L2), 0.92-0.96 (L6-15), ratio 1.03-1.09; residual mismatch ~3e-6 rms at the final layer

## Programs

- [code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v3.py](code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v3.py)
- [code/opus_reexam_20261001_v1/engineer/stack/build.py](code/opus_reexam_20261001_v1/engineer/stack/build.py)
- [code/opus_reexam_20261001_v1/engineer/stack/s48_build.sh](code/opus_reexam_20261001_v1/engineer/stack/s48_build.sh)
- [code/opus_reexam_20261001_v1/engineer/a52_pld/dly_from_pt_v2.py](code/opus_reexam_20261001_v1/engineer/a52_pld/dly_from_pt_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a48_parity/parity_change.py](code/opus_reexam_20261001_v1/engineer/a48_parity/parity_change.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a10_bd32/fragment_bd32.py](code/opus_reexam_20261001_v1/engineer/a10_bd32/fragment_bd32.py)
- [code/opus_reexam_20261001_v1/engineer/a11_bare_views/analyze_binds.py](code/opus_reexam_20261001_v1/engineer/a11_bare_views/analyze_binds.py)
- [code/opus_reexam_20261001_v1/engineer/a11_bare_views/fragment_bvp.py](code/opus_reexam_20261001_v1/engineer/a11_bare_views/fragment_bvp.py)
- [code/opus_reexam_20261001_v1/engineer/a12_live_slice/fragment_dcr.py](code/opus_reexam_20261001_v1/engineer/a12_live_slice/fragment_dcr.py)
- [code/opus_reexam_20261001_v1/engineer/a17_tanh/fragment_ets.py](code/opus_reexam_20261001_v1/engineer/a17_tanh/fragment_ets.py)
- [code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut.py](code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut.py)

<details>
<summary>Additional programs and dependencies (41)</summary>

- [code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut_v2.py](code/opus_reexam_20261001_v1/engineer/a19_edge_cut/fragment_ecut_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3.py](code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3.py)
- [code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3o.py](code/opus_reexam_20261001_v1/engineer/a21_load3/fragment_load3o.py)
- [code/opus_reexam_20261001_v1/engineer/a29_e2/fragment_e2_v1.py](code/opus_reexam_20261001_v1/engineer/a29_e2/fragment_e2_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a31_star/fragment_sx_v1.py](code/opus_reexam_20261001_v1/engineer/a31_star/fragment_sx_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a32_c2k/fragment_c2k_v1.py](code/opus_reexam_20261001_v1/engineer/a32_c2k/fragment_c2k_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/fragment_ss_v3.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/fragment_ss_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a35_join/fragment_jr_v1.py](code/opus_reexam_20261001_v1/engineer/a35_join/fragment_jr_v1.py)
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
- [code/opus_reexam_20261001_v1/engineer/a47_kc/fragment_kc_v1.py](code/opus_reexam_20261001_v1/engineer/a47_kc/fragment_kc_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a49_band/fragment_band_v1.py](code/opus_reexam_20261001_v1/engineer/a49_band/fragment_band_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a50_b7/fragment_b7_v2.py](code/opus_reexam_20261001_v1/engineer/a50_b7/fragment_b7_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v2.py](code/opus_reexam_20261001_v1/engineer/a52_pld/fragment_pld_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a54_jmiv/read_pt.py](code/opus_reexam_20261001_v1/engineer/a54_jmiv/read_pt.py)
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

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1066](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1066)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1067](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1067)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1068](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1068)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1069](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1069)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-11-per-layer-carried-column-scale-hook-byte-identical-identity) · [Copy provenance](../../PROVENANCE.json)

