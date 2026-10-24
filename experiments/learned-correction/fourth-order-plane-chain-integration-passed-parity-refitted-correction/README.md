# Fourth-order plane-chain integration passed parity; its refitted correction remained worse than baseline

2026-09-04 · experiment · informational · measured

THE CHAIN'S OWN k4 DIAGONAL IS NOT BETTER THAN THE CARD'S (r35, 32M truth): chain k4 slope 1.00-1.06, corr 0.992 (L2) -> 0.942 (L14), rel err 13% -> 34%; the derived card's provider k4 slope 1.08-1.25, corr 0.994 -> 0.965, rel err 13% -> 33%. The kurtosis shortfall on the derived host is per-unit structure, not scale (consistent with T-88h's dead table). STATUS for Matt (13:50): nothing below 2e-8 raw on the bench; best deployable object = the derived card (2.18e-8 / 3.00e-8, util ~0.82-0.85, adjusted ~2.1e-8 vs the bank 1.78e-8): a raw-only improvement; the remaining error is per-unit v/k3/k4 structure (half-truth -70%) with no computable supply at the needed alignment.

Recorded measurement:

    - T-89 IN-CARD FOURTH-ORDER PLANE CHAIN, CERTIFIED (public_g29_bond_kernel_v1/k22_chain.py = the engine's fourth_emission_1024 + plane-supported push without the 1024^4 term, wired into estimator_retro.py as RETRO_K22_CHAIN=1; seeded at layer 0 like the engine; carries its own k4/K22/K31, fed by the card's live m, v, C, K21, k3; pre_k22 := chain K22): r35 chain K22 vs the engine's deploy-frame K22 corr 0.9998-0.9999, ratio 1.001 (L4) .. 1.028 (L14), identical truth corr (0.985/0.974/0.958/0.946 at L4/8/12/14); host F2J off 3.092e-8 = the forced-engine-K22 host. No tensor-engine run needed per net any more: rows, panels and virgin gates of the derived host are one card run each. (Derived-host k4 table, T-88h: half-scale 2.146e-8 (-1.4%), full-scale 2.278e-8 / 3.143e-8 - nothing.)
    
    - T-89b REFIT ROWS ON THE DERIVED HOST v1 (in-card chain, no feed / carry): gen_rows_k22chain.py (copy of the k4panel row generator with RETRO_K22_CHAIN=1, no forcing; host row bit-identical to the certified chain host on r35; 49 s per row), run_gen_k22chain.py driving ranks 35-234 two at a time (~80 min) -> rows_k22chain/. Next: run_refit_ext.py with G22_ROWS_DIR=rows_k22chain (frozen 160/40 protocol) against the old host's 3.13e-8 held-out under the same protocol.
    
    - T-89c REFIT ON THE DERIVED HOST v1 (in-card chain only; rows_k22chain 35-234; frozen 160/40 protocol; REFIT_K22CHAIN_RESULT.json): valid-40 final 3.228e-8 vs its host 3.590e-8 = 10.1% removal (the same protocol removes 37.7% on the old host -> 3.134e-8); independent 4M: r35 refit 2.880e-8 (host 3.092e-8, -6.9%; +6.5% vs the deployed full card), r100 2.905e-8 (host 3.708e-8, -22%; +4.5% vs deployed). THE REFIT DOOR IS CLOSED WITH A CLEAN TEST: on a host whose fourth-order plane defect is fixed, the tape predicts only ~10% of what is left; the 688-row recipe could add at most the 6-8% protocol gap. The derived host's remaining error is per-unit state content the tape does not carry (half-truth v/k3/k4 -70%). Best raw object on the bench stays the derived host v2 (chain + feed + carry, no operator): r35 2.18e-8 / r100 3.00e-8 at util ~0.82.
    
    - T-89d THE DERIVED HOST'S SKEW IS ENGINE-CLASS WITH NO SCALE BIAS (bank dumps, 32M truth): per-unit k3 rel err 3.0-5.8% at L2-14 on r35 and 3.0-5.6% on r100, corr 0.998-1.000, truth-on-derived slope 1.000-1.014 at every layer (the deployed card: 5-11%, the engine 3-7%). No k3 table lever on this host (tables k3_scale_derived_*.json written, not worth running). Its remaining per-unit errors are incoherent (k3) or structural (k4 8-23% under, v).
    
    - T-89e THE CHAIN'S OWN k4 DIAGONAL IS NOT BETTER THAN THE CARD'S (r35, 32M truth): chain k4 slope 1.00-1.06, corr 0.992 (L2) -> 0.942 (L14), rel err 13% -> 34%; the derived card's provider k4 slope 1.08-1.25, corr 0.994 -> 0.965, rel err 13% -> 33%. The kurtosis shortfall on the derived host is per-unit structure, not scale (consistent with T-88h's dead table). STATUS for Matt (13:50): nothing below 2e-8 raw on the bench; best deployable object = the derived card (2.18e-8 / 3.00e-8, util ~0.82-0.85, adjusted ~2.1e-8 vs the bank 1.78e-8): a raw-only improvement; the remaining error is per-unit v/k3/k4 structure (half-truth -70%) with no computable supply at the needed alignment.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g22_operator_features_v1/gen_rows_k22chain.py](code/public_g22_operator_features_v1/gen_rows_k22chain.py)
- [code/public_g22_operator_features_v1/run_refit_ext.py](code/public_g22_operator_features_v1/run_refit_ext.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)
- [code/public_g22_operator_features_v1/run_gen_k22chain.py](code/public_g22_operator_features_v1/run_gen_k22chain.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/gen_rows_clcarried.py](code/public_g22_operator_features_v1/gen_rows_clcarried.py)
- [code/public_g22_operator_features_v1/gen_rows_closedloop.py](code/public_g22_operator_features_v1/gen_rows_closedloop.py)
- [code/public_g22_operator_features_v1/gen_rows_meanleak.py](code/public_g22_operator_features_v1/gen_rows_meanleak.py)
- [code/public_g22_operator_features_v1/gen_rows_mixed.py](code/public_g22_operator_features_v1/gen_rows_mixed.py)
- [code/public_g22_operator_features_v1/gen_rows_mixed025.py](code/public_g22_operator_features_v1/gen_rows_mixed025.py)
- [code/public_g22_operator_features_v1/jext/run_experiment_ext.py](code/public_g22_operator_features_v1/jext/run_experiment_ext.py)
- [code/public_g22_operator_features_v1/run_gen_clcarried.py](code/public_g22_operator_features_v1/run_gen_clcarried.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/public_g22_operator_features_v1/run_gen_closedloop.py](code/public_g22_operator_features_v1/run_gen_closedloop.py)
- [code/public_g22_operator_features_v1/run_gen_meanleak.py](code/public_g22_operator_features_v1/run_gen_meanleak.py)
- [code/public_g22_operator_features_v1/run_gen_mixed.py](code/public_g22_operator_features_v1/run_gen_mixed.py)
- [code/public_g22_operator_features_v1/run_gen_mixed025.py](code/public_g22_operator_features_v1/run_gen_mixed025.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)

</details>

## Evidence

- [LANE_g3_tracker.md#L754](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L754)
- [LANE_g3_tracker.md#L755](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L755)
- [LANE_g3_tracker.md#L756](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L756)
- [LANE_g3_tracker.md#L757](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L757)
- [LANE_g3_tracker.md#L758](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L758)

<details>
<summary>All result files and records (2)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [REFIT_K22CHAIN_RESULT.json](REFIT_K22CHAIN_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-fourth-order-plane-chain-integration-passed-parity-refitted-correction) · [Copy provenance](../../../PROVENANCE.json)

