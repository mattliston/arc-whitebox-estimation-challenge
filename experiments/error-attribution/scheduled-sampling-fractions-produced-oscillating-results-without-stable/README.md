# Scheduled-sampling fractions produced oscillating results without stable improvement

2026-09-04 · experiment · informational · measured

IN-LOOP MECHANISTIC FEATURES wired (RETRO_CL_MECH=1 with RETRO_DUMP_FEAT=1: the 31 G28 features computed online from the layer's dump via mechfeat_from_run and appended to the tape row before _cl_step; final layer zero columns as in the rows). Used only with a 51-input payload; to be verified against rows_meanleak_mech/row_0035/mech.npy when the MEANLEAK_MECH refit lands (armed).

Code availability: The mixed-row generators, refitter and retrofit instrument were recovered. The cited entry also describes newly wired mechanistic features still awaiting verification; it does not establish that verification completed at this point.

Recorded measurement:

    - T-113 SCHEDULED SAMPLING OSCILLATES (closed-loop F2J-off, r35 / r100): alpha 1 (oracle rows) 3.28e-8 / 3.73e-8; alpha 0.5 2.87e-8 / 4.55e-8; alpha 0.25 3.62e-8 / 3.59e-8; alpha 0 (pure DAgger) 5.82e-8 / 1.05e-7; open-loop deployed 2.70e-8 / 2.78e-8. Removal on the training rows falls with alpha (60 -> 54.5 -> 47.9 -> 33%) because the targets carry more of the unobservable transported residual; the loop never approaches the teacher-forced 8.9e-9. Confirms T-110/T-112: the binding constraint is the fresh per-layer removal r (60% at L8-15 with 20 tape features); scheduled sampling only redistributes it. ROOT-LEVER TEST running: the oracle rows with the 31 G28 mechanistic features (rows_meanleak_mech, 4 workers) -> 51-feature teacher-forced refit; if r rises materially, the in-loop features must be computed online (mechfeat per layer inside _cl_step) and the loop re-run.
    
    - T-113b IN-LOOP MECHANISTIC FEATURES wired (RETRO_CL_MECH=1 with RETRO_DUMP_FEAT=1: the 31 G28 features computed online from the layer's dump via mechfeat_from_run and appended to the tape row before _cl_step; final layer zero columns as in the rows). Used only with a 51-input payload; to be verified against rows_meanleak_mech/row_0035/mech.npy when the MEANLEAK_MECH refit lands (armed).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/gen_rows_mixed.py](code/public_g22_operator_features_v1/gen_rows_mixed.py)
- [code/public_g22_operator_features_v1/gen_rows_mixed025.py](code/public_g22_operator_features_v1/gen_rows_mixed025.py)
- [code/public_g22_operator_features_v1/run_refit_generic.py](code/public_g22_operator_features_v1/run_refit_generic.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L804](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L804)
- [LANE_g3_tracker.md#L805](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L805)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-scheduled-sampling-fractions-produced-oscillating-results-without-stable) · [Copy provenance](../../../PROVENANCE.json)

