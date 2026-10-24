# Untrained online residual correction diverged under every tested damping and depth window

2026-09-04 · experiment · informational · measured

closed loop with the existing (open-loop-trained) operator at any damping/window is worse than open loop (deployed 2.70e-8): window 14-15 5.20e-8, 12-15 1.87e-7, 10-15 3.16e-7, mix 0.25 full 4.46e-8, mix 0.5 window 12-15 4.10e-8. PLAN T-105 (teacher forcing at the HOST level, Keenan's mechanism without backprop): rows from the oracle recursion (true post means fed back with the consistent covariance; the row's host = the ONE-STEP mean before the correction; targets = the organizers' 1e9 all-layer means, which exist for the 688 nets incl. ranks 35-234) -> refit the operator on those rows (frozen 160/40 protocol) -> export the payload -> closed-loop inference (its own corrections fed back). Exposure bias handled by a second iteration (rows from the closed loop of the first operator).

Code availability: The shared closed-loop correction and retrofit implementations were recovered. The tracker combines measured divergence with a proposed teacher-forcing follow-up; it does not certify that follow-up at this point.

Recorded measurement:

    - T-104 CLOSED LOOP WITH THE UNTRAINED OPERATOR DIVERGES (RETRO_CLOSED_LOOP=1: _cl_step = transport_correction stepped online; its correction carried into mean + consistent covariance): r35 4.25e-7, r100 6.41e-7, mean-only 6.45e-7, mix 0.5 8.72e-8 (= host). The operator was trained open-loop on the uncorrected host's tape; fed back it over-corrects and compounds. Keenan's fix: train through the recursion. Cheap alternatives: a late-window closed loop, damping, or the ITERATED open-loop retrain (closed-loop rows -> refit -> repeat) with the existing rows/refit machinery.
    
    - T-104b closed loop with the existing (open-loop-trained) operator at any damping/window is worse than open loop (deployed 2.70e-8): window 14-15 5.20e-8, 12-15 1.87e-7, 10-15 3.16e-7, mix 0.25 full 4.46e-8, mix 0.5 window 12-15 4.10e-8. PLAN T-105 (teacher forcing at the HOST level, Keenan's mechanism without backprop): rows from the oracle recursion (true post means fed back with the consistent covariance; the row's host = the ONE-STEP mean before the correction; targets = the organizers' 1e9 all-layer means, which exist for the 688 nets incl. ranks 35-234) -> refit the operator on those rows (frozen 160/40 protocol) -> export the payload -> closed-loop inference (its own corrections fed back). Exposure bias handled by a second iteration (rows from the closed loop of the first operator).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/closedloop_layers.py](code/public_g22_operator_features_v1/closedloop_layers.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L784](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L784)
- [LANE_g3_tracker.md#L785](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L785)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-untrained-online-residual-correction-diverged-under-every-tested-damping) · [Copy provenance](../../../PROVENANCE.json)

