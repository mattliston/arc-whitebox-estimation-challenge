# Native refit parity: prediction-change correlation +0.897, relative size 1.16 and FLOPs +0.006%

2026-10-12T05:12:44Z · finding · informational · measured

The native candidate generation 50 (S50) file (jmS50_v1, 10209ba7) against native S48b on 7100521, compared with the trainer's PyTorch records (a_RS50C_pt vs a_RS50C_ab): the final-layer prediction change correlates +0.897 with the PyTorch change at native / PyTorch size 1.16; metered FLOPs +0.006% (244.035e9 vs 244.020e9 flops), +78 operations; final-layer MSE native -0.107% vs PyTorch +0.368% on this one network (layers 8-14 native -0.14..-0.68% vs PyTorch +0.06..-0.39%, same sign or near zero, native a little stronger). It became the positive control of the route-parity rule and certified the files submitted as #335505 / #335508.

Recorded measurement:

    corr +0.897, native / torch size 1.16, changes 5.50e-6 torch / 6.40e-6 native of the prediction; bill +0.006% (244.035e9 vs 244.020e9), +78 operations; final MSE native -0.107% vs torch +0.368% on 7100521

## Programs

- [../2026-10-12-swap-growth-1-58-2-54-gb-kills-certificates/candidate.py](../2026-10-12-swap-growth-1-58-2-54-gb-kills-certificates/candidate.py)
- [../2026-10-12-swap-growth-1-58-2-54-gb-kills-certificates/chain_s50par_v1.sh](../2026-10-12-swap-growth-1-58-2-54-gb-kills-certificates/chain_s50par_v1.sh)
- [../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](../../../instruments/native40-scorer-seals-85-calls-requires-lower-own-bill/code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [../2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together/gate_rdz_v1.py](../2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together/gate_rdz_v1.py)
- [../2026-10-12-swap-growth-1-58-2-54-gb-kills-certificates/parity_v2.py](../2026-10-12-swap-growth-1-58-2-54-gb-kills-certificates/parity_v2.py)
- [../2026-10-11-plane-read-column-cut-2-0-adopted-about-0/candidate.py](../2026-10-11-plane-read-column-cut-2-0-adopted-about-0/candidate.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1354](../../TEAM_LOG_2026-10-01_to_10-22.md#L1354)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1399](../../TEAM_LOG_2026-10-01_to_10-22.md#L1399)

<details>
<summary>All result files and records (65)</summary>

- [a_RS50C_pt_r7100532.json](a_RS50C_pt_r7100532.json) · result
- [a_RS50C_pt_r7100524.json](a_RS50C_pt_r7100524.json) · result
- [a_RS50C_pt_r7100549.json](a_RS50C_pt_r7100549.json) · result
- [a_RS50C_pt_r7100528.json](a_RS50C_pt_r7100528.json) · result
- [a_RS50C_pt_r7100545.json](a_RS50C_pt_r7100545.json) · result
- [a_RS50C_pt_r7100544.json](a_RS50C_pt_r7100544.json) · result
- [a_RS50C_pt_r7100529.json](a_RS50C_pt_r7100529.json) · result
- [a_RS50C_pt_r7100548.json](a_RS50C_pt_r7100548.json) · result
- [a_RS50C_pt_r7100525.json](a_RS50C_pt_r7100525.json) · result
- [a_RS50C_pt_r7100533.json](a_RS50C_pt_r7100533.json) · result
- [a_RS50C_pt_r7100538.json](a_RS50C_pt_r7100538.json) · result
- [a_RS50C_pt_r7100543.json](a_RS50C_pt_r7100543.json) · result
- [a_RS50C_pt_r7100534.json](a_RS50C_pt_r7100534.json) · result
- [a_RS50C_pt_r7100522.json](a_RS50C_pt_r7100522.json) · result
- [a_RS50C_pt_r7100523.json](a_RS50C_pt_r7100523.json) · result
- [a_RS50C_pt_r7100535.json](a_RS50C_pt_r7100535.json) · result
- [a_RS50C_pt_r7100542.json](a_RS50C_pt_r7100542.json) · result
- [a_RS50C_pt_r7100539.json](a_RS50C_pt_r7100539.json) · result
- [a_RS50C_pt_r7100541.json](a_RS50C_pt_r7100541.json) · result
- [a_RS50C_pt_r7100520.json](a_RS50C_pt_r7100520.json) · result
- [a_RS50C_pt_r7100536.json](a_RS50C_pt_r7100536.json) · result
- [a_RS50C_pt_r7100537.json](a_RS50C_pt_r7100537.json) · result
- [a_RS50C_pt_r7100521.json](a_RS50C_pt_r7100521.json) · result
- [a_RS50C_pt_r7100540.json](a_RS50C_pt_r7100540.json) · result
- [a_RS50C_pt_r7100526.json](a_RS50C_pt_r7100526.json) · result
- [a_RS50C_pt_r7100530.json](a_RS50C_pt_r7100530.json) · result
- [a_RS50C_pt_r7100547.json](a_RS50C_pt_r7100547.json) · result
- [a_RS50C_pt_r7100551.json](a_RS50C_pt_r7100551.json) · result
- [a_RS50C_pt_r7100550.json](a_RS50C_pt_r7100550.json) · result
- [a_RS50C_pt_r7100546.json](a_RS50C_pt_r7100546.json) · result
- [a_RS50C_pt_r7100531.json](a_RS50C_pt_r7100531.json) · result
- [a_RS50C_pt_r7100527.json](a_RS50C_pt_r7100527.json) · result
- [a_RS50C_ab_r7100540.json](a_RS50C_ab_r7100540.json) · result
- [a_RS50C_ab_r7100521.json](a_RS50C_ab_r7100521.json) · result
- [a_RS50C_ab_r7100537.json](a_RS50C_ab_r7100537.json) · result
- [a_RS50C_ab_r7100536.json](a_RS50C_ab_r7100536.json) · result
- [a_RS50C_ab_r7100520.json](a_RS50C_ab_r7100520.json) · result
- [a_RS50C_ab_r7100541.json](a_RS50C_ab_r7100541.json) · result
- [a_RS50C_ab_r7100527.json](a_RS50C_ab_r7100527.json) · result
- [a_RS50C_ab_r7100531.json](a_RS50C_ab_r7100531.json) · result
- [a_RS50C_ab_r7100546.json](a_RS50C_ab_r7100546.json) · result
- [a_RS50C_ab_r7100550.json](a_RS50C_ab_r7100550.json) · result
- [a_RS50C_ab_r7100551.json](a_RS50C_ab_r7100551.json) · result
- [a_RS50C_ab_r7100547.json](a_RS50C_ab_r7100547.json) · result
- [a_RS50C_ab_r7100530.json](a_RS50C_ab_r7100530.json) · result
- [a_RS50C_ab_r7100526.json](a_RS50C_ab_r7100526.json) · result
- [a_RS50C_ab_r7100533.json](a_RS50C_ab_r7100533.json) · result
- [a_RS50C_ab_r7100525.json](a_RS50C_ab_r7100525.json) · result
- [a_RS50C_ab_r7100548.json](a_RS50C_ab_r7100548.json) · result
- [a_RS50C_ab_r7100529.json](a_RS50C_ab_r7100529.json) · result
- [a_RS50C_ab_r7100544.json](a_RS50C_ab_r7100544.json) · result
- [a_RS50C_ab_r7100545.json](a_RS50C_ab_r7100545.json) · result
- [a_RS50C_ab_r7100528.json](a_RS50C_ab_r7100528.json) · result
- [a_RS50C_ab_r7100549.json](a_RS50C_ab_r7100549.json) · result
- [a_RS50C_ab_r7100524.json](a_RS50C_ab_r7100524.json) · result
- [a_RS50C_ab_r7100532.json](a_RS50C_ab_r7100532.json) · result
- [a_RS50C_ab_r7100539.json](a_RS50C_ab_r7100539.json) · result
- [a_RS50C_ab_r7100542.json](a_RS50C_ab_r7100542.json) · result
- [a_RS50C_ab_r7100535.json](a_RS50C_ab_r7100535.json) · result
- [a_RS50C_ab_r7100523.json](a_RS50C_ab_r7100523.json) · result
- [a_RS50C_ab_r7100522.json](a_RS50C_ab_r7100522.json) · result
- [a_RS50C_ab_r7100534.json](a_RS50C_ab_r7100534.json) · result
- [a_RS50C_ab_r7100543.json](a_RS50C_ab_r7100543.json) · result
- [a_RS50C_ab_r7100538.json](a_RS50C_ab_r7100538.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-native-refit-parity-prediction-change-correlation-0-897-relative) · [Copy provenance](../../../PROVENANCE.json)

