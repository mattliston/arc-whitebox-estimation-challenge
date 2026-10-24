# Native-route table updates automatically from certified parity verdicts

2026-10-12T16:56:06Z · tool · adopted · measured

The band-dial rule (b) stopped hard-coding families: trainer/cert/refit-after-drop (RAD)/ROUTES.json lists per-layer families (iso, nfs, nkl1-3, k4la1, kc, s_carry, then k21n / rsy after their parity passes), all-band-only families (birth scale coefficient (bscale), birth-leg coefficient (X1), ssb, nfd, n6b, n6r, n6d) and k4's bands 1-7 / 8-11 / 12-14 / 15 / all (the native _S19_K4BAND + _K4_15), is read by every round's make file and rewritten by routes_update when an engineer line beginning 'ROUTE PARITY <family>_<layer>: PASSED' appears. For radlab's strongest single-layer gradients the engineer prepared routes: bscale by layer and X1 by layer (the builder's 10-10 DLY edits, no-op c3ee7c1c), k22 by layer (scaling the scalar radial on the k22 line, zero extra flops) and x2a1 by birth layer (F4C's strengths folded into the newborn's table, so layer 7's arm = birth 6); the dry run of rule (b) on candidate generation 51 (S51)'s gradients selected iso 11-14 (t -2.83), k21n 7-10 (-3.14), k4 15-15 (+4.77) and rsy 11-14 (-4.03).

Recorded measurement:

    dry run on S51's 64 gradients: iso 11-14 t -2.83, k21n 7-10 -3.14, k4 15-15 +4.77 (halves +3.49 / +3.25), rsy 11-14 -4.03, k4's other bands |t| < 1; radlab single-layer A36 signals bscale L0 -5.3, k4 L15 +4.8, k22 L14 +3.4, x2a1 L7 -3.3, X1 L14 -2.9 (fresh t = 0.54 x A36 t, corr 0.89); x2a_from_pt on arm_px2a1_7_7.pt = {6: ((0.0, 0.509128468809), (0.0, 0.258157425747))}

## Programs

- [code/opus_reexam_20261001_v1/trainer/cert/RAD/make_routes_table_v1.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/make_routes_table_v1.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/routes_update_v1.sh](code/opus_reexam_20261001_v1/trainer/cert/RAD/routes_update_v1.sh)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/routes_update_v3.sh](code/opus_reexam_20261001_v1/trainer/cert/RAD/routes_update_v3.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_pld_bx_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_pld_bx_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_k22_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_k22_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/k22_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/k22_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_x2a_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_x2a_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/x2a_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/x2a_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/chain_routes_v2.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/chain_routes_v2.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/chain_x2a_v1.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/chain_x2a_v1.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/parity_verdict_v3.py](code/opus_reexam_20261001_v1/engineer/a62_modules/parity_verdict_v3.py)
- [code/opus_reexam_20261001_v1/trainer/export_dials_s46_v7.py](code/opus_reexam_20261001_v1/trainer/export_dials_s46_v7.py)

<details>
<summary>Additional programs and dependencies (11)</summary>

- [code/opus_reexam_20261001_v1/engineer/a52_pld/dly_from_pt_v3.py](code/opus_reexam_20261001_v1/engineer/a52_pld/dly_from_pt_v3.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/engineer/a54_jmiv/read_pt.py](code/opus_reexam_20261001_v1/engineer/a54_jmiv/read_pt.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/gate_rdz_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/gate_rdz_v1.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v2.py](code/opus_reexam_20261001_v1/release/guard_footprint_v2.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/routes_update_v1.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/routes_update_v1.py)
- [code/opus_reexam_20261001_v1/trainer/cert/RAD/routes_update_v3.py](code/opus_reexam_20261001_v1/trainer/cert/RAD/routes_update_v3.py)
- [code/opus_reexam_20261001_v1/trainer/export.py](code/opus_reexam_20261001_v1/trainer/export.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1543](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1543)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1545](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1545)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1552](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1552)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1553](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1553)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1555](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1555)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1557](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1557)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1568](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1568)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1569](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1569)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1571](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1571)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1585](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1585)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1599](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1599)

<details>
<summary>All result files and records (2)</summary>

- [ROUTES.json](ROUTES.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-12-native-route-table-updates-automatically-certified-parity-verdicts) · [Copy provenance](../../PROVENANCE.json)

