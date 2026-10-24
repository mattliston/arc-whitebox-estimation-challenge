# Half-teacher scheduled sampling changed training error without a transferable endpoint gain

2026-09-04 · experiment · informational · measured

CARRIED-ERROR PROFILES IN THE LOOP (organizers' truth; closedloop_layers.py profile mode): operator_1 r35 fourth-cumulant Edgeworth operator (L4) 4.1e-9 / L8 1.47e-8 / L12 3.27e-8 / L15 3.16e-8, mixed05 3.95e-9 / 1.31e-8 / 2.94e-8 / 3.13e-8; r100 operator_1 3.7e-9 / 1.31e-8 / 2.33e-8 / 3.41e-8, mixed05 3.6e-9 / 1.34e-8 / 2.68e-8 / 4.11e-8. The two loops agree through L10 and split at L12-15: the exposure gap is a last-layers phenomenon on both nets (T-108a again). Sub-problem: the operator's corrections at L13-14 under its own upstream; the second stage (open-loop patch on the carried host) targets exactly the accumulated late residual. (The alpha 0.25 drivers were relaunched after a script-name slip.)

Recorded measurement:

    - T-111 SCHEDULED SAMPLING, STEP 1 (alpha 0.5; REFIT_MIXED05): valid-40 1.126e-8 vs its one-step host 2.475e-8 = 54.5% removal (oracle rows 60%, pure-loop rows 33%); on its training distribution r35 9.56e-9 / r100 1.20e-8. CLOSED LOOP (alpha 0 at inference): r35 2.87e-8 (operator_1 3.28e-8; open-loop deployed 2.70e-8), r100 4.55e-8 (operator_1 3.73e-8). Progress on r35, regression on r100: the inference residual is still 2x the training residual. Continuing the schedule: alpha 0.25 rows from the mixed operator's loop -> refit -> loop. In parallel the second stage (open-loop patch refit on the closed loop's carried means, rows_clcarried) is generating.
    
    - T-111b CARRIED-ERROR PROFILES IN THE LOOP (organizers' truth; closedloop_layers.py profile mode): operator_1 r35 L4 4.1e-9 / L8 1.47e-8 / L12 3.27e-8 / L15 3.16e-8, mixed05 3.95e-9 / 1.31e-8 / 2.94e-8 / 3.13e-8; r100 operator_1 3.7e-9 / 1.31e-8 / 2.33e-8 / 3.41e-8, mixed05 3.6e-9 / 1.34e-8 / 2.68e-8 / 4.11e-8. The two loops agree through L10 and split at L12-15: the exposure gap is a last-layers phenomenon on both nets (T-108a again). Sub-problem: the operator's corrections at L13-14 under its own upstream; the second stage (open-loop patch on the carried host) targets exactly the accumulated late residual. (The alpha 0.25 drivers were relaunched after a script-name slip.)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g22_operator_features_v1/closedloop_layers.py](code/public_g22_operator_features_v1/closedloop_layers.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/closedloop_layers_gnn.py](code/public_g22_operator_features_v1/closedloop_layers_gnn.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L800](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L800)
- [LANE_g3_tracker.md#L801](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L801)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-half-teacher-scheduled-sampling-changed-training-error-without-transferable) · [Copy provenance](../../../PROVENANCE.json)

