# Second on-policy fit diverged; mixed teacher and model rows bounded the intermediate error

2026-09-04 · experiment · informational · measured

SCHEDULED-SAMPLING ROWS ARE SANE (r35, alpha 0.5 truth share, operator_1): one-step error fourth-cumulant Edgeworth operator (L4) 4.4e-9 / L8 1.42e-8 / L12 2.46e-8 / L15 2.47e-8 - between the oracle (4.1e-9 / 1.33e-8 / 2.31e-8 / 2.25e-8) and the pure loop (5.8e-9 / 2.07e-8 / 4.56e-8 / 4.51e-8), close to the oracle. (An earlier divergence was the oracle hook firing alongside the blend; RETRO_MEANLEAK_OFF=1 now guards it.) COVARIANCE-CONSISTENCY LAW (T-108 diagnostic 4): operator means with a TRUTH-consistent covariance diverge (2.9e-4): the covariance update must use exactly the delta carried in the mean; consistent updates with an imperfect delta are stable, inconsistent ones explode - the exposure gap cannot be split by channel. rows_mixed generating (two drivers).

Code availability: The closed-loop and mixed-row generators, generic refitter and retrofit instrument were recovered. A dedicated launch script for each diagnostic summarized by this tracker entry was not identified.

Recorded measurement:

    - T-109 ITERATION 2 DIVERGED (REFIT_MEANLEAK_IT2: rows = operator_1's closed loop): valid-40 3.37e-8 vs its one-step host 5.01e-8 = 32.9% removal (60% on the oracle upstream); closed loop with operator_2: r35 5.82e-8, r100 1.05e-7 (operator_1: 3.28e-8 / 3.73e-8). CAUSE (T-108 diagnostics 1-3): in the loop the one-step error = fresh part (60% predictable) + the transported residual the operator missed upstream, and that residual is unobservable from the tape (removal 30% at every layer in-loop, T-108b; the ladder puts the gap in the last three layers, T-108a); pure DAgger makes each round's target less predictable -> divergence. ATTACK: scheduled sampling - rows with the fed-back correction = alpha x true delta + (1-alpha) x operator delta (RETRO_CLOSED_LOOP_EXPOSURE), alpha 0.5 first, then decreasing; the operator learns on intermediate distributions instead of the worst one.
    
    - T-109b SCHEDULED-SAMPLING ROWS ARE SANE (r35, alpha 0.5 truth share, operator_1): one-step error L4 4.4e-9 / L8 1.42e-8 / L12 2.46e-8 / L15 2.47e-8 - between the oracle (4.1e-9 / 1.33e-8 / 2.31e-8 / 2.25e-8) and the pure loop (5.8e-9 / 2.07e-8 / 4.56e-8 / 4.51e-8), close to the oracle. (An earlier divergence was the oracle hook firing alongside the blend; RETRO_MEANLEAK_OFF=1 now guards it.) COVARIANCE-CONSISTENCY LAW (T-108 diagnostic 4): operator means with a TRUTH-consistent covariance diverge (2.9e-4): the covariance update must use exactly the delta carried in the mean; consistent updates with an imperfect delta are stable, inconsistent ones explode - the exposure gap cannot be split by channel. rows_mixed generating (two drivers).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/gen_rows_closedloop.py](code/public_g22_operator_features_v1/gen_rows_closedloop.py)
- [code/public_g22_operator_features_v1/gen_rows_mixed.py](code/public_g22_operator_features_v1/gen_rows_mixed.py)
- [code/public_g22_operator_features_v1/run_refit_generic.py](code/public_g22_operator_features_v1/run_refit_generic.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L796](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L796)
- [LANE_g3_tracker.md#L797](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L797)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-second-policy-fit-diverged-mixed-teacher-model-rows-bounded) · [Copy provenance](../../../PROVENANCE.json)

