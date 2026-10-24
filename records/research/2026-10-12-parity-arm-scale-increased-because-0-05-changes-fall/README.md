# Parity-arm scale increased because +0.05 changes fall below the 2e-6 prediction floor

2026-10-12T22:41:00Z · finding · informational · measured

The trainer's candidate generation 50 (S50) single-layer PyTorch arms at +0.05 moved the final prediction by 1.7e-7 (birth scale coefficient (bscale) at L0), 5.4e-7 / 6.0e-7 (birth-leg coefficient (X1) at L14) and 2.3e-7 / 2.4e-7 (x2a1 at L7) of its size, 3.5-12x under the parity rule's 2e-6 floor while the native-vs-PyTorch mismatch of passed arms is about 1-2e-6, so those routes could only come out not comparable or fail on noise (k22 x1.05 at L14, 5.9e-6 / 6.4e-6, was judgeable); larger arms (bscale +1.5 at L0, x2a1 +1.0 at L7) moved the prediction 3.55e-6 / 3.42e-6 and 3.92e-6 / 3.94e-6, a 17-21x response for a 20-30x shift (mildly sublinear), and were judged; X1 at L14 was dropped.

Recorded measurement:

    +0.05 arms: bscale 1.7e-7; X1 5.4e-7 / 6.0e-7; x2a1 2.3e-7 / 2.4e-7; k22 x1.05 5.9e-6 / 6.4e-6 (torch final MSE +0.67% / +0.93%); v2 arms: bscale +1.5 3.55e-6 / 3.42e-6, x2a1 +1.0 3.92e-6 / 3.94e-6; response 17-21x for 20-30x the shift

## Programs

- [chain_routes_v5.sh](chain_routes_v5.sh)
- [chain_x2a_v3.sh](chain_x2a_v3.sh)
- [parity_verdict_v4.py](parity_verdict_v4.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [../2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together/gate_rdz_v1.py](../2026-10-11-six-native-jobs-trip-swap-guard-five-terminate-together/gate_rdz_v1.py)
- [../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_pld_bx_v1.py](../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_pld_bx_v1.py)
- [../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_x2a_v1.py](../../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/make_x2a_v1.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/release/guard_footprint_v2.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/release/guard_footprint_v2.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1707](../../TEAM_LOG_2026-10-01_to_10-22.md#L1707)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1741](../../TEAM_LOG_2026-10-01_to_10-22.md#L1741)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1744](../../TEAM_LOG_2026-10-01_to_10-22.md#L1744)

<details>
<summary>All result files and records (5)</summary>

- [a_RS50_pbscale15_0_0_r7100456.json](a_RS50_pbscale15_0_0_r7100456.json) · result
- [a_RS50_pbscale15_0_0_r7100457.json](a_RS50_pbscale15_0_0_r7100457.json) · result
- [a_RS50_px2a1p1_7_7_r7100456.json](a_RS50_px2a1p1_7_7_r7100456.json) · result
- [a_RS50_px2a1p1_7_7_r7100457.json](a_RS50_px2a1p1_7_7_r7100457.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-parity-arm-scale-increased-because-0-05-changes-fall) · [Copy provenance](../../../PROVENANCE.json)

