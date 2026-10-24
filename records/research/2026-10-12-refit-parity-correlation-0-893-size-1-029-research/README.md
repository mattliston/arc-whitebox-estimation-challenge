# Refit parity: correlation +0.893, size 1.029; research MSE -0.530% versus native -0.155%

2026-10-12T20:18:01Z · finding · informational · measured

The port loop's parity for candidate generation 51 (S51) on the check block's first seed 7100648 (native S51 vs the native candidate generation 50 (S50) safe jmS50_v1_pbs, against the trainer's a_RS51C records): final-layer prediction change correlation +0.893 at native / PyTorch norm 1.029, while the final-layer MSE change was PyTorch -0.530% vs native -0.155% on that network. Identity checks: safe + own prebind list vs safe byte-identical (metered FLOPs -0.0035%, ops -58); twin + own list + scatter-based gather (XSG) vs the twin source byte-identical (metered FLOPs -0.0786%, ops +203); counts 43,133 / 41,448 (safe) and 45,061 / 43,418 (twin) normal.

Recorded measurement:

    7100648: L15 corr +0.893, norm native/torch 1.029, MSE change torch -0.530% native -0.155%; safe identity bill -0.0035%, ops -58; twin identity bill -0.0786%, ops +203; counts s51pbs 43,133 / 41,448 normal, 43,492 / 41,540 slow; s51twx 45,061 / 43,418 normal, 45,424 / 43,468 slow

## Programs

- [candidate.py](candidate.py)
- [candidate-f314d8ea.py](candidate-f314d8ea.py)
- [../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/chain_rad_port_v9.sh](../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/chain_rad_port_v9.sh)
- [port_rad_v1n.sh](port_rad_v1n.sh)
- [port_twin_v3.sh](port_twin_v3.sh)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers/read_pt.py](../2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers/read_pt.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/edx_from_pt_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/edx_from_pt_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_early_edge_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_early_edge_v1.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k21n_v1.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k21n_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k3l_bylayer_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k3l_bylayer_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4_v1.py)

<details>
<summary>Additional programs and dependencies (29)</summary>

- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4la2_v1.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_k4la2_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_prebind_s48b_v2.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_prebind_s48b_v2.py)
- [../2026-10-11-plane-read-column-cut-2-0-adopted-about-0/fragment_readcol_v1.py](../2026-10-11-plane-read-column-cut-2-0-adopted-about-0/fragment_readcol_v1.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_rs_v1.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_rs_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_beta_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/fragment_star_beta_v1.py)
- [../2026-10-11-one-direction-covariance-star-adopted-0-26-mse-0/fragment_star_dir_v1.py](../2026-10-11-one-direction-covariance-star-adopted-0-26-mse-0/fragment_star_dir_v1.py)
- [../2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together/gate_rdz_v1.py](../2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together/gate_rdz_v1.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/k21n_from_pt_v2.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/k21n_from_pt_v2.py)
- [../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/k22_from_pt_v1.py](../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/k22_from_pt_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/k3l_from_pt_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/k3l_from_pt_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/k4_from_pt_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/k4_from_pt_v1.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/k4la2_from_pt_v1.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/k4la2_from_pt_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/kappa_from_pt_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/kappa_from_pt_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/law_literals_from_pt_v2.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/law_literals_from_pt_v2.py)
- [../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_k22_v1.py](../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_k22_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/make_kappa_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/make_kappa_v1.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py)
- [../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_x2a_v1.py](../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_x2a_v1.py)
- [../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_xsg_any_v1.py](../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_xsg_any_v1.py)
- [../2026-10-12-swap-growth-1-58-2-54-gb-kills-certificates/parity_v2.py](../2026-10-12-swap-growth-1-58-2-54-gb-kills-certificates/parity_v2.py)
- [../../../instruments/targeted-prebinding-replaces-unused-keys-first-network-requests-fall/code/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py](../../../instruments/targeted-prebinding-replaces-unused-keys-first-network-requests-fall/code/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py)
- [pt_families_v8.py](pt_families_v8.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/rs_from_pt_v1.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/engineer/a62_modules/rs_from_pt_v1.py)
- [../2026-10-11-plane-read-column-cut-2-0-adopted-about-0/candidate.py](../2026-10-11-plane-read-column-cut-2-0-adopted-about-0/candidate.py)
- [../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py](../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/star_beta_from_pt_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/star_beta_from_pt_v1.py)
- [../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/x2a_from_pt_v1.py](../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/x2a_from_pt_v1.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/release/guard_footprint_v2.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/release/guard_footprint_v2.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1666](../../TEAM_LOG_2026-10-01_to_10-22.md#L1666)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1667](../../TEAM_LOG_2026-10-01_to_10-22.md#L1667)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-refit-parity-correlation-0-893-size-1-029-research) · [Copy provenance](../../../PROVENANCE.json)

