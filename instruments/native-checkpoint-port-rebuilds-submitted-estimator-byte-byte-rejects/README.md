# Native checkpoint port rebuilds the submitted estimator byte for byte and rejects unrouted families

2026-10-11T19:31:02Z · tool · adopted · measured

The port script (a62_modules/port_rad_v1b.sh through v1i.sh) turns a trainer hand-over plus its params.pt into a release-ready native file through the standard build (law literals checked against the checkpoint, release fragments, star strength beta, read column cut, prebind list); rehearsed on the carrier-strength refit-after-drop round (MS) hand-over it reproduced the submitted candidate generation 48 (S48) candidate bit for bit on 7100328. A family check (pt_families v1-v4) stops the port on any non-zero refit family without a certified native route; routes were added for the early edge (edx), per-layer cohort-age (3,1) plane term (k31l), the k4 band plus a final-layer radial k4 hook (fragment_k4_v1: layers 0-14 bit for bit at 0, +2.24% final MSE for a 10% scale) and the star strength beta (beta 1 = the shipped form bit for bit), while kappa was dropped after its route failed parity. Two defects were caught: v1 picked its source by mtime and took another base's file (caught by the rehearsal), and v1h stopped on a 259-byte source file name built from 17-digit lever values (v1i writes 12 significant digits, relative change < 1e-12).

Recorded measurement:

    rehearsal on 7100328: prediction sha 5f09aa4a = S48 bit for bit, bill 249.0877 G vs 249.9906 G (-0.36%, the prebind on an in-process first network), 37,095 vs 37,998 operations; k4c0 bit for bit, k4c1 (_K4_15 0.1) layers 0-14 bit for bit, final MSE +2.24%, +3 operations; k3l ages 1-2 at 1.0 at read layers 5-9: layers 0-4 bit for bit, final +0.48%, -10 operations; star beta 1 = ss -1 bit for bit, beta 1.5 +0.26% on 7100136

## Programs

- [code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1b.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1b.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1c.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1c.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1d.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1d.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1e.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1e.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1f.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1f.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1g.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1g.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1h.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1h.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1i.sh](code/opus_reexam_20261001_v1/engineer/a62_modules/port_rad_v1i.sh)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/pt_families_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/pt_families_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_beta_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_beta_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k3l_bylayer_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k3l_bylayer_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/k3l_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/k3l_from_pt_v1.py)

<details>
<summary>Additional programs and dependencies (25)</summary>

- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/k4_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/k4_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/edx_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/edx_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_early_edge_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_early_edge_v1.py)
- [code/opus_reexam_20261001_v1/trainer/export_dials.py](code/opus_reexam_20261001_v1/trainer/export_dials.py)
- [code/opus_reexam_20261001_v1/engineer/stack/s48_build.sh](code/opus_reexam_20261001_v1/engineer/stack/s48_build.sh)
- [code/opus_reexam_20261001_v1/engineer/stack/release_like.py](code/opus_reexam_20261001_v1/engineer/stack/release_like.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a52_pld/dly_from_pt_v2.py](code/opus_reexam_20261001_v1/engineer/a52_pld/dly_from_pt_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a54_jmiv/read_pt.py](code/opus_reexam_20261001_v1/engineer/a54_jmiv/read_pt.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_prebind_s48b_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_prebind_s48b_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_readcol_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_readcol_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_dir_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_dir_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/kappa_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/kappa_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/law_literals_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/law_literals_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/law_literals_from_pt_v2.py](code/opus_reexam_20261001_v1/engineer/a62_modules/law_literals_from_pt_v2.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_kappa_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_kappa_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/pt_families_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/pt_families_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/pt_families_v3.py](code/opus_reexam_20261001_v1/engineer/a62_modules/pt_families_v3.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/pt_families_v4.py](code/opus_reexam_20261001_v1/engineer/a62_modules/pt_families_v4.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s48/s48b/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s48/s48b/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/star_beta_from_pt_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/star_beta_from_pt_v1.py)
- [code/opus_reexam_20261001_v1/trainer/export.py](code/opus_reexam_20261001_v1/trainer/export.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1176](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1176)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1177](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1177)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1200](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1200)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1211](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1211)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1213](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1213)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1238](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1238)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1240](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1240)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1246](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1246)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1305](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1305)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1327](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1327)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-11-native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects) · [Copy provenance](../../PROVENANCE.json)

