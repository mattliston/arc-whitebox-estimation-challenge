# Second-stage correction removed only 8.0% of the closed-loop host error

2026-09-04 · experiment · informational · measured

rows_mixed025 complete (200); the MIXED025 refit + loop chain is running. Building rows_meanleak_mech: the oracle recursion with the 31 G28 mechanistic features per row (gen_rows_meanleak_mech.py, derived from round 3's gen_rows_mech.py) for a 51-feature refit (G22_LOCAL_FEATURES=51) of the teacher-forced operator - the root-lever test (per-layer fresh removal 60% -> ?).

Recorded measurement:

    - T-112 SECOND STAGE DEAD (REFIT_CLCARRIED: open-loop patch on operator_1's closed-loop carried host): valid-40 3.415e-8 vs host 3.710e-8 = 8.0% removal; r35 3.28e-8 -> 3.00e-8, r100 3.73e-8 -> 3.57e-8. The closed loop's accumulated residual is not tape-predictable (the original host's 65%-predictable error was the systematic fourth-order plane defect; what the loop leaves is the unpredictable 40% of each layer's fresh error, transported). ROOT LEVER: the per-layer fresh removal r (60% teacher-forced at L8-15); the loop's late error ~ (1-r)^2 -> r 0.6 -> 0.75 would cut it ~2.4x. The one-step error is a LOCAL function of the card's state, so the lever is richer late-layer inputs (the 31 mechanistic features of G28 lifted local dk3 predictability 0.4 -> 0.6): refit the oracle operator with mechanistic inputs.
    
    - T-112b rows_mixed025 complete (200); the MIXED025 refit + loop chain is running. Building rows_meanleak_mech: the oracle recursion with the 31 G28 mechanistic features per row (gen_rows_meanleak_mech.py, derived from round 3's gen_rows_mech.py) for a 51-feature refit (G22_LOCAL_FEATURES=51) of the teacher-forced operator - the root-lever test (per-layer fresh removal 60% -> ?).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech.py)
- [code/public_g22_operator_features_v1/gen_rows_mech.py](code/public_g22_operator_features_v1/gen_rows_mech.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/gen_rows_force14_test.py](code/public_g22_operator_features_v1/gen_rows_force14_test.py)
- [code/public_g22_operator_features_v1/gen_rows_gnnmixed.py](code/public_g22_operator_features_v1/gen_rows_gnnmixed.py)
- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech15.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech15.py)
- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py)
- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_k22.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_k22.py)
- [code/public_g22_operator_features_v1/run_gen_meanleak_mech.py](code/public_g22_operator_features_v1/run_gen_meanleak_mech.py)
- [code/public_g22_operator_features_v1/run_gen_mech.py](code/public_g22_operator_features_v1/run_gen_mech.py)
- [code/public_g28_mech_features_v1/mechfeat.py](code/public_g28_mech_features_v1/mechfeat.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

</details>

## Evidence

- [LANE_g3_tracker.md#L802](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L802)
- [LANE_g3_tracker.md#L803](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L803)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-second-stage-correction-removed-only-8-0-closed-loop) · [Copy provenance](../../../PROVENANCE.json)

