# Estimator builder composes implementation fragments and checks tensor contractions

2026-10-01T22:18Z · tool · informational · measured

The engineer's stack builder takes a parent estimator source and a list of lever names (float32 fixes, strips, prebind lists, physics fragments, request levers, release fragments) and writes a candidate file; every build is checked by an einsum site diff against the bank for new or repeated-operand signatures. All seven files submitted in this slice were built this way, the later ones on top of trainer exports.

Recorded measurement:

    builds S36 (sha 87b3d6ef), S37 (2a25a074), the S38-S42 release payloads

## Programs

- [code/engineer/stack/build.py](code/engineer/stack/build.py)
- [code/engineer/stack/levers_from_params.py](code/engineer/stack/levers_from_params.py)
- [code/engineer/a10_bd32/fragment_bd32.py](code/engineer/a10_bd32/fragment_bd32.py)
- [code/engineer/a11_bare_views/analyze_binds.py](code/engineer/a11_bare_views/analyze_binds.py)
- [code/engineer/a11_bare_views/fragment_bvp.py](code/engineer/a11_bare_views/fragment_bvp.py)
- [code/engineer/a12_live_slice/fragment_dcr.py](code/engineer/a12_live_slice/fragment_dcr.py)
- [code/engineer/a17_tanh/fragment_ets.py](code/engineer/a17_tanh/fragment_ets.py)
- [code/engineer/a21_load3/fragment_load3.py](code/engineer/a21_load3/fragment_load3.py)
- [code/engineer/a21_load3/fragment_load3o.py](code/engineer/a21_load3/fragment_load3o.py)
- [code/engineer/a29_e2/fragment_e2_v1.py](code/engineer/a29_e2/fragment_e2_v1.py)
- [code/engineer/a31_star/fragment_sx_v1.py](code/engineer/a31_star/fragment_sx_v1.py)
- [code/engineer/a32_c2k/fragment_c2k_v1.py](code/engineer/a32_c2k/fragment_c2k_v1.py)

<details>
<summary>Additional programs and dependencies (41)</summary>

- [code/engineer/a33_starsum/fragment_ss_v3.py](code/engineer/a33_starsum/fragment_ss_v3.py)
- [code/engineer/a35_join/fragment_jr_v1.py](code/engineer/a35_join/fragment_jr_v1.py)
- [code/engineer/a40_setup/fragment_cs_v1.py](code/engineer/a40_setup/fragment_cs_v1.py)
- [code/engineer/a41_base/fragment_bk3_v1.py](code/engineer/a41_base/fragment_bk3_v1.py)
- [code/engineer/a41_base/fragment_bsk_v1.py](code/engineer/a41_base/fragment_bsk_v1.py)
- [code/engineer/a41_base/fragment_gsk_v1.py](code/engineer/a41_base/fragment_gsk_v1.py)
- [code/engineer/a41_base/fragment_so2_v1.py](code/engineer/a41_base/fragment_so2_v1.py)
- [code/engineer/a41_base/fragment_so2_v2.py](code/engineer/a41_base/fragment_so2_v2.py)
- [code/engineer/a42_head/fragment_hc_v1.py](code/engineer/a42_head/fragment_hc_v1.py)
- [code/engineer/a42_head/fragment_hc_v2.py](code/engineer/a42_head/fragment_hc_v2.py)
- [code/engineer/a43_readout/fragment_ro_v1.py](code/engineer/a43_readout/fragment_ro_v1.py)
- [code/engineer/a45_l15/fragment_l15_v1.py](code/engineer/a45_l15/fragment_l15_v1.py)
- [code/engineer/a47_kc/fragment_kc_v1.py](code/engineer/a47_kc/fragment_kc_v1.py)
- [code/engineer/a49_band/fragment_band_v1.py](code/engineer/a49_band/fragment_band_v1.py)
- [code/engineer/a50_b7/fragment_b7_v2.py](code/engineer/a50_b7/fragment_b7_v2.py)
- [code/engineer/a55_u15/fragment_u15_v1.py](code/engineer/a55_u15/fragment_u15_v1.py)
- [code/engineer/a56_req/fragment_rqs_v1.py](code/engineer/a56_req/fragment_rqs_v1.py)
- [code/engineer/a60_psdm/fragment_psdm_v1.py](code/engineer/a60_psdm/fragment_psdm_v1.py)
- [code/engineer/buildlib.py](code/engineer/buildlib.py)
- [code/engineer/edits_lib.py](code/engineer/edits_lib.py)
- [code/engineer/hzs/fragment_gen_v7.py](code/engineer/hzs/fragment_gen_v7.py)
- [code/ncar.py](code/ncar.py)
- [code/ncar4.py](code/ncar4.py)
- [code/ncar5.py](code/ncar5.py)
- [code/physicist/b6cols.py](code/physicist/b6cols.py)
- [code/physicist/b7cols.py](code/physicist/b7cols.py)
- [code/physicist/f4b.py](code/physicist/f4b.py)
- [code/physicist/ncar6_audit_frozen_s3d.py](code/physicist/ncar6_audit_frozen_s3d.py)
- [code/physicist/ncar_bank_frozen_s2s.py](code/physicist/ncar_bank_frozen_s2s.py)
- [code/physicist/ncar_fold_frozen_s2w.py](code/physicist/ncar_fold_frozen_s2w.py)
- [code/physicist/ncarry.py](code/physicist/ncarry.py)
- [code/physicist/pbase_frozen_s4a.py](code/physicist/pbase_frozen_s4a.py)
- [code/trainer/frozen/k31l_t1508.py](code/trainer/frozen/k31l_t1508.py)
- [code/trainer/frozen/k31l_t1508y.py](code/trainer/frozen/k31l_t1508y.py)
- [code/trainer/frozen/ncar6_t1339.py](code/trainer/frozen/ncar6_t1339.py)
- [code/trainer/frozen/ncar6_t1402.py](code/trainer/frozen/ncar6_t1402.py)
- [code/trainer/frozen/ncar6_t1402x.py](code/trainer/frozen/ncar6_t1402x.py)
- [code/trainer/frozen/ncar6_t1402y.py](code/trainer/frozen/ncar6_t1402y.py)
- [code/trainer/otr_common.py](code/trainer/otr_common.py)
- [code/trainer/otr_m.py](code/trainer/otr_m.py)
- [code/trainer/otr_pl.py](code/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L43](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L43)
- [TEAM_LOG_2026-10-01_to_10-22.md#L66](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L66)
- [TEAM_LOG_2026-10-01_to_10-22.md#L99](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L99)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-01-estimator-builder-composes-implementation-fragments-checks-tensor-contractions) · [Copy provenance](../../PROVENANCE.json)

