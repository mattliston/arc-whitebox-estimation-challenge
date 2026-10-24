# Native per-layer routes added for kink-plane scale, young/old reads and fourth-order age-2 strength

2026-10-12T07:39:36Z · tool · adopted · measured

Before the generated rounds could dial per-layer families, the engineer built text-edit routes with bit-for-bit no-ops: fragment_k21n_v1 (the kink third-cumulant plane with index multiplicities (2,1) (K21) knob by layer through _pld_knobs, with the exporter's per-layer shift replacing the knob, no double count), fragment_rs_v1 (read scales lam x (1 + rso) on the old tier's columns and x (1 + rsy) on the young ones at the read only) and fragment_k4la2_v1 (the cohort-age local fourth-order term (K4L) age-2 strength at the layer the term is applied: K4L v7 swaps _K4L_C's age-2 entry inside _k4l_local at layer l, K4L v6 rescales the cohort born at j by value(j + 2) / _K4L_C[1] at birth). The round rules then fixed which families may vary per layer (iso, nfs, nkl1-3, k4la1, kc, s_carry; k21n / rsy only after their parity PASS) and which stay one literal (birth scale coefficient (bscale), birth-leg coefficient (X1), ssb, k4, nfd, n6b, n6r, n6d), and the trainer's exporter v5 / v6 learned to write per-layer k4la2 and s_carry.

Recorded measurement:

    no-ops bit for bit: k21n0s50 and rs0s50 on 7100521 vs fb277a8a (prediction cd478a30fbbe56d9, bill 244,031,024,124, 36,857 operations); fragment_k4la2 on the S50 safe (K4L v6) and S50 twin f04fa779 (K4L v7, pred 5422571864f114b8, 258,782,500,820 flops, 39,704 ops) with the empty table and with the file's own literal; parity diso corr +0.892 / +0.832 size 1.13 / 1.14, dk21n +0.837 / +0.850 size 1.11 / 1.20, drsy +0.976 / +0.987 size 1.01

## Programs

- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k21n_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k21n_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/k21n_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/k21n_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/k21n_from_pt_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/k21n_from_pt_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_rs_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_rs_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/rs_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/rs_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4la2_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4la2_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/k4la2_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/k4la2_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/pt_families_v6.py](code/opus_reexam_20261001_v1/engineer/a62_modules/pt_families_v6.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1l.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1l.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/parity_verdict_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/parity_verdict_v2.py)
- [code/opus_reexam_20261001_v1/trainer/export_dials_s46_v5.py](code/opus_reexam_20261001_v1/trainer/export_dials_s46_v5.py)
- [code/opus_reexam_20261001_v1/trainer/export_dials_s46_v6.py](code/opus_reexam_20261001_v1/trainer/export_dials_s46_v6.py)

<details>
<summary>Additional programs and dependencies (33)</summary>

- [code/opus_reexam_20261001_v1/trainer/cert/RAD/S53/make_s53_v2.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/S53/make_s53_v2.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/S53/make_s53_v3.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/S53/make_s53_v3.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/dryrun/k4la2_band_v5/candidate.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/dryrun/k4la2_band_v5/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/chain_s52parity_v1.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/chain_s52parity_v1.sh)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/engineer/a54_jmiv/read_pt.py](code/opus_reexam_20261001_v1/engineer/a54_jmiv/read_pt.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/edx_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/edx_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_early_edge_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_early_edge_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k3l_bylayer_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k3l_bylayer_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_prebind_s48b_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_prebind_s48b_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_readcol_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_readcol_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_beta_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_beta_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_dir_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_dir_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/gate_rdz_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/gate_rdz_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/k3l_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/k3l_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/k4_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/k4_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/kappa_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/kappa_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/law_literals_from_pt_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/law_literals_from_pt_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_kappa_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_kappa_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s48/s48b/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s48/s48b/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/star_beta_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/star_beta_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v2.py](code/opus_reexam_20261001_v1/release/guard_footprint_v2.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py)
- [code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_v7.py](code/opus_reexam_20261001_v1/trainer/cert/refit/jmix_generated_v7.py)
- [code/opus_reexam_20261001_v1/trainer/export.py](code/opus_reexam_20261001_v1/trainer/export.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py](code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1322](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1322)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1392](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1392)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1397](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1397)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1402](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1402)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1403](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1403)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1407](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1407)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1413](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1413)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1414](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1414)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1420](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1420)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1442](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1442)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1448](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1448)

<details>
<summary>All result files and records (6)</summary>

- [README.txt](README.txt) · result
- [head_weights_31x128.json](head_weights_31x128.json) · result
- [hu.json](hu.json) · result
- [per_layer_dials.json](per_layer_dials.json) · result
- [physics_dials.json](physics_dials.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-12-native-per-layer-routes-added-kink-plane-scale-young) · [Copy provenance](../../PROVENANCE.json)

