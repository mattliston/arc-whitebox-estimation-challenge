# Gaussian-closure training required covariance repair and still removed only about 60–64% of error

2026-09-04 · experiment · informational · measured

ROW-HEALTH AUDIT (one-step error profile L10 -> L15 over 12 nets): HEALTHY (flat): mech15b (full card), nq (no-quartic), cap (fourth-cumulant canonical-polyadic carrier (CP4) 256), lean2, gausspsd. EXPLODED (indefinite consistent covariance update): lean (97x), diag (562x), cp3low (65x), gauss (unprojected, 500x). Teacher-forced removal numbers on the exploded sets (HDIAG, HCP3LOW, Hlean, HGAUSS 90.7%) are VOID; the no-quartic 87% and LEAN-2 numbers stand.

Recorded measurement:

    - T-123 NEW-FAMILY TEST 1 (GPU only, no card runs): the competitor's util 0.22 ~ the Gaussian card (0.16-0.18) + ~0.04 = the cost of a ~1M-parameter per-unit corrector per layer, not a tensor bank. Our 64-channel closure reaches 90.7% teacher-forced on the Gaussian card's rows (host 3.11e-4); the bar at util 0.18 needs ~98% in the loop. Does capacity move it? Train 256-channel / 48-message closures on rows_meanleak_mech15_gauss (existing) and read the held-out removal curve (train_gnn_hgauss_c256.log).
    
    - T-123 CAVEAT: the Gaussian card's ORACLE rows are unstable at the last layers (one-step error L1-L10 1.6-5e-7 = the true Gaussian fresh error, then L12 1.5e-6, L14 9.4e-6, L15 2.4e-4: the consistent covariance update with the Gaussian card's large mean corrections blows the covariance up). The 90.7% "final removal" and the loop's 1.2e-6 are on/against corrupted late-layer targets; judge the capacity test by the INTERIOR layers (L8/L12: 46% / 22% at 64ch). A future Gaussian-card row set needs a damped consistent update (RETRO_MEANLEAK_MIX < 1 or no delta-delta term) - only if the family continues.
    
    - T-123 FIRST SIGNAL: on the Gaussian card's rows, a closure with 48 MESSAGE channels (x3 edge powers) and 256 hidden (128 state channels, from scratch) removes L8 54% / L12 56% / L14 66% at epoch 5, vs the 64-channel baseline 46 / 22 / 8 and the 256-state-channel run 45 / 20 / 12 (epoch 5). The width of the messages through W, W^2, W^3 - the capacity to carry the third-order HISTORY - is the binding dimension, not the per-unit state. Next: message width 128 / two message hops per layer; stable Gaussian-card rows (damped consistent update) for honest final-layer targets; then the loop at util 0.18 (bar: raw < ~5.6e-8 grader).
    
    - T-123b: the Gaussian card's oracle-recursion instability is the CONSISTENT COVARIANCE UPDATE going indefinite (min eigenvalue -1.5e-3 at L2, -0.16 at L11, -2.2 at L14: the rank-2 mixed term -(m d^T + d m^T) has eigenvalues +-|m||d| and the card's covariance has no margin). PSD projection after the update (RETRO_MEANLEAK_PSD=1) fixes it: r35 final one-step 3.4e-4 -> 6.2e-7 (= the interior level 4-7e-7). Generating rows_meanleak_mech15_gausspsd (200 nets, 6 workers) for honest late-layer targets. Message-width curve at epoch 20: 48 msg ch -> L8 57 / L12 68 / L14 77%; 256 state ch -> 53 / 25 / 18%; 64 ch baseline 46 / 22 / 8%.
    
    - T-123c: fnp parity of the in-loop step at the new widths (48 msg ch, 256 hidden, 128 state): max |fnp - torch|/rms 8e-7..2e-6 per layer (float32 level); the closure's own bill 1.56e10 flops = 0.71% of the budget (0.21% at 64 ch). Parity gotcha: gnn_parity.py's torch stage builds the model from the env, so GNN_FIRST/GNN_CM/GNN_HIDDEN must be set (without GNN_FIRST=1 the torch model starts at layer 2 and the "mismatch" is 27x rms at L2 - not a real failure). Message width 48 vs 128 at epoch 45: 59/71/83 vs 60/73/83 (L8/12/14): saturated at 48. The state-width run (256 ch, 16 msg) was stopped at epoch 35: 55/26/24.
    
    - T-123 VERDICT (new family test, honest): on STABILISED Gaussian-card rows (PSD-projected consistent update; one-step final host 5.9e-7, smooth profile), the 48-message closure removes 60-64% at EVERY layer (peak 64.3% at epoch 20, overfits to 58% by 60; L8/12/14 = 62/62/64) - the same ~62-65% information ceiling as the full card's closure. The 83%-at-L14 "message-width signal" was an artifact of the exploded late-layer targets (the explosion is a predictable function of the state). Closed loop with this closure on the Gaussian card: r35 1.32e-5 / r100 1.56e-5 at util 0.160 (DIVERGES vs the card's open loop 2.8e-6 and the 64-ch loop 1.2e-6: state mismatch between the projected rows and the unprojected loop). Bar needed ~98% removal. FAMILY CLOSED: a per-unit learned closure has a ~62-65% ceiling on any moment card, cheap or rich. The "competitor = Gaussian card + learned corrector" hypothesis is falsified for this corrector class.
    
    - T-123 ROW-HEALTH AUDIT (one-step error profile L10 -> L15 over 12 nets): HEALTHY (flat): mech15b (full card), nq (no-quartic), cap (CP4 256), lean2, gausspsd. EXPLODED (indefinite consistent covariance update): lean (97x), diag (562x), cp3low (65x), gauss (unprojected, 500x). Teacher-forced removal numbers on the exploded sets (HDIAG, HCP3LOW, Hlean, HGAUSS 90.7%) are VOID; the no-quartic 87% and LEAN-2 numbers stand. Standing habit: every new oracle row set gets this profile check before any training is read.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g22_operator_features_v1/gnn_parity.py](code/public_g22_operator_features_v1/gnn_parity.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L912](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L912)
- [LANE_g3_tracker.md#L913](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L913)
- [LANE_g3_tracker.md#L914](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L914)
- [LANE_g3_tracker.md#L915](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L915)
- [LANE_g3_tracker.md#L916](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L916)
- [LANE_g3_tracker.md#L917](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L917)
- [LANE_g3_tracker.md#L918](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L918)

<details>
<summary>All result files and records (2)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [train_gnn_hgauss_c256.log](train_gnn_hgauss_c256.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-gaussian-closure-training-required-covariance-repair-still-removed-only) · [Copy provenance](../../../PROVENANCE.json)

