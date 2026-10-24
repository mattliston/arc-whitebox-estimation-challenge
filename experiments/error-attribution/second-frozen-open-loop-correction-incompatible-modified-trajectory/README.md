# A second frozen open-loop correction was incompatible with the modified trajectory

2026-09-04 · experiment · informational · measured

the DEPLOYED open-loop patch on top of operator_1's closed loop is off-distribution and hurts (r35 6.35e-8, r100 1.13e-7 vs the loop alone 3.28e-8 / 3.73e-8) - the second stage must be refit on the closed-loop host's carried means (rows_clcarried: r35 accumulated residual fourth-cumulant Edgeworth operator (L4) 4.1e-9, L8 1.47e-8, L12 3.27e-8, L15 3.16e-8; generating for ranks 35-234). Two refits queued: MIXED (scheduled sampling, alpha 0.5) and CLCARRIED (second-stage open-loop patch).

Code availability: The carried-row generator, generic refitter and retrofit instrument were recovered. The tracker explicitly says two refits were queued; their source availability does not establish completion at this point.

Recorded measurement:

    - T-110 THE 30% DECOMPOSED: teacher-forced per-layer removal of operator_1 is 24% (L2) -> 60% (L8-15) (REFIT_MEANLEAK_RESULT all-200 layer MSEs vs the oracle one-step host); in its own loop the one-step error = fresh part (~oracle level) + the TRANSPORTED residual of what it missed upstream (about equal at depth), and it removes ~60% of the fresh part and ~0% of the transported part -> 30% overall, exactly as observed (T-108b). Without its recurrent state the loop is worse (6.2e-8 vs 3.28e-8), so the state helps; the wall is the transported residual, which by construction is absent from the oracle rows and unpredictable from local tape in the loop rows (why DAgger diverged, T-109). CHAIN OF IDEAS: the OPEN-loop operator's whole skill is transporting an accumulated error estimate (65% at the final layer). Two-stage design: closed-loop operator_1 corrects the fresh births in the recursion; a second, open-loop operator refit on the closed-loop host's CARRIED means (targets = its accumulated residual) patches the output rows. Testing: (a) the deployed open-loop patch on top of the closed loop as-is; (b) rows_clcarried -> refit -> patch.
    
    - T-110b the DEPLOYED open-loop patch on top of operator_1's closed loop is off-distribution and hurts (r35 6.35e-8, r100 1.13e-7 vs the loop alone 3.28e-8 / 3.73e-8) - the second stage must be refit on the closed-loop host's carried means (rows_clcarried: r35 accumulated residual L4 4.1e-9, L8 1.47e-8, L12 3.27e-8, L15 3.16e-8; generating for ranks 35-234). Two refits queued: MIXED (scheduled sampling, alpha 0.5) and CLCARRIED (second-stage open-loop patch).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/gen_rows_clcarried.py](code/public_g22_operator_features_v1/gen_rows_clcarried.py)
- [code/public_g22_operator_features_v1/run_refit_generic.py](code/public_g22_operator_features_v1/run_refit_generic.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L798](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L798)
- [LANE_g3_tracker.md#L799](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L799)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-second-frozen-open-loop-correction-incompatible-modified-trajectory) · [Copy provenance](../../../PROVENANCE.json)

