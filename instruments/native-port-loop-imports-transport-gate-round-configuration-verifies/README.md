# Native port loop imports the transport gate from round configuration and verifies its presence

2026-10-12T22:28:58Z · tool · adopted · measured

After gk11 was absorbed into the refit rounds from candidate generation 53 (S53), the port loop (chain_rad_port_v10 / v11) reads each round's trainer/cert/refit-after-drop (RAD)/S<N>/config.json patch.ncar through gk_from_config_v1.py (the default carrier list plus estimator physics switch set (pbase) gsk / gk4 / gbk only; any other carrier change stops the round before any build), cross-checks it against the round's PyTorch records, applies the skew- and kurtosis-corrected transport gate (gk) text edit (make_gsk_v1.py) last on the safe and twin files, uses a start file with gk as the parity base, and from v11 runs gk_guard_v1.py at three points (G1 the config's strengths equal the absorbed 1.0 / 1.0 / 0.0 for every round from the absorb round on; G2 the built sources and G3 the finals carry the gk edit with those strengths), posting an anchored stopped line on any failure.

Recorded measurement:

    26/26 scratch checks: strength 0 = jmS51_v1 / jmS51_v1tw byte for byte, gk 1.0 = the first stack's s51k1 files byte for byte; guard tests 14/14; S52 reads GK NONE and ports as before

## Programs

- [code/engineer/a62_modules/chain_rad_port_v10.sh](code/engineer/a62_modules/chain_rad_port_v10.sh)
- [code/engineer/a62_modules/chain_rad_port_v11.sh](code/engineer/a62_modules/chain_rad_port_v11.sh)
- [code/engineer/a62_modules/gk_from_config_v1.py](code/engineer/a62_modules/gk_from_config_v1.py)
- [code/engineer/a62_modules/gk_guard_v1.py](code/engineer/a62_modules/gk_guard_v1.py)
- [code/engineer/a62_modules/make_gsk_v1.py](code/engineer/a62_modules/make_gsk_v1.py)
- [code/engineer/a62_modules/port_rad_v1o.sh](code/engineer/a62_modules/port_rad_v1o.sh)
- [code/engineer/a62_modules/port_twin_v4.sh](code/engineer/a62_modules/port_twin_v4.sh)
- [code/engineer/a62_modules/preflight_gk_v2.sh](code/engineer/a62_modules/preflight_gk_v2.sh)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a41_base/fragment_gsk_v1.py](code/engineer/a41_base/fragment_gsk_v1.py)
- [code/engineer/a54_jmiv/read_pt.py](code/engineer/a54_jmiv/read_pt.py)
- [code/engineer/a62_modules/edx_from_pt_v1.py](code/engineer/a62_modules/edx_from_pt_v1.py)

<details>
<summary>Additional programs and dependencies (33)</summary>

- [code/engineer/a62_modules/fragment_early_edge_v1.py](code/engineer/a62_modules/fragment_early_edge_v1.py)
- [code/engineer/a62_modules/fragment_k21n_v1.py](code/engineer/a62_modules/fragment_k21n_v1.py)
- [code/engineer/a62_modules/fragment_k3l_bylayer_v1.py](code/engineer/a62_modules/fragment_k3l_bylayer_v1.py)
- [code/engineer/a62_modules/fragment_k4_v1.py](code/engineer/a62_modules/fragment_k4_v1.py)
- [code/engineer/a62_modules/fragment_k4la2_v1.py](code/engineer/a62_modules/fragment_k4la2_v1.py)
- [code/engineer/a62_modules/fragment_prebind_s48b_v2.py](code/engineer/a62_modules/fragment_prebind_s48b_v2.py)
- [code/engineer/a62_modules/fragment_readcol_v1.py](code/engineer/a62_modules/fragment_readcol_v1.py)
- [code/engineer/a62_modules/fragment_rs_v1.py](code/engineer/a62_modules/fragment_rs_v1.py)
- [code/engineer/a62_modules/fragment_star_beta_v1.py](code/engineer/a62_modules/fragment_star_beta_v1.py)
- [code/engineer/a62_modules/fragment_star_dir_v1.py](code/engineer/a62_modules/fragment_star_dir_v1.py)
- [code/engineer/a62_modules/gate_rdz_v1.py](code/engineer/a62_modules/gate_rdz_v1.py)
- [code/engineer/a62_modules/k21n_from_pt_v2.py](code/engineer/a62_modules/k21n_from_pt_v2.py)
- [code/engineer/a62_modules/k22_from_pt_v1.py](code/engineer/a62_modules/k22_from_pt_v1.py)
- [code/engineer/a62_modules/k3l_from_pt_v1.py](code/engineer/a62_modules/k3l_from_pt_v1.py)
- [code/engineer/a62_modules/k4_from_pt_v1.py](code/engineer/a62_modules/k4_from_pt_v1.py)
- [code/engineer/a62_modules/k4la2_from_pt_v1.py](code/engineer/a62_modules/k4la2_from_pt_v1.py)
- [code/engineer/a62_modules/kappa_from_pt_v1.py](code/engineer/a62_modules/kappa_from_pt_v1.py)
- [code/engineer/a62_modules/law_literals_from_pt_v2.py](code/engineer/a62_modules/law_literals_from_pt_v2.py)
- [code/engineer/a62_modules/make_k22_v1.py](code/engineer/a62_modules/make_k22_v1.py)
- [code/engineer/a62_modules/make_kappa_v1.py](code/engineer/a62_modules/make_kappa_v1.py)
- [code/engineer/a62_modules/make_s49_v1.py](code/engineer/a62_modules/make_s49_v1.py)
- [code/engineer/a62_modules/make_x2a_v1.py](code/engineer/a62_modules/make_x2a_v1.py)
- [code/engineer/a62_modules/make_xsg_any_v1.py](code/engineer/a62_modules/make_xsg_any_v1.py)
- [code/engineer/a62_modules/parity_v2.py](code/engineer/a62_modules/parity_v2.py)
- [code/engineer/a62_modules/prebind_targeted_v2.py](code/engineer/a62_modules/prebind_targeted_v2.py)
- [code/engineer/a62_modules/pt_families_v8.py](code/engineer/a62_modules/pt_families_v8.py)
- [code/engineer/a62_modules/rs_from_pt_v1.py](code/engineer/a62_modules/rs_from_pt_v1.py)
- [code/engineer/a62_modules/s48/s48b/candidate.py](code/engineer/a62_modules/s48/s48b/candidate.py)
- [code/engineer/a62_modules/s49/xsgF/candidate.py](code/engineer/a62_modules/s49/xsgF/candidate.py)
- [code/engineer/a62_modules/star_beta_from_pt_v1.py](code/engineer/a62_modules/star_beta_from_pt_v1.py)
- [code/engineer/a62_modules/x2a_from_pt_v1.py](code/engineer/a62_modules/x2a_from_pt_v1.py)
- [code/release/guard_footprint_v2.py](code/release/guard_footprint_v2.py)
- [code/release/rusage.py](code/release/rusage.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1697](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1697)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1730](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1730)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1731](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1731)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1738](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1738)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-12-native-port-loop-imports-transport-gate-round-configuration-verifies) · [Copy provenance](../../PROVENANCE.json)

