# Age-dependent low-rank source cohorts were emulated inside the estimator

2026-09-04 · experiment · informational · measured

LOW-RANK COHORT BANK, IN-CARD EMULATION (RETRO_LOWRANK_CP3 = "P ranks by age;Q ranks by age": after each push every cohort block is replaced by its rank-truncated SVD, so raw is exactly what a low-rank card gives; independent 4M; deployed r35 2.7038e-8 on / 8.622e-8 off, r100 2.7787e-8 / 7.696e-8): schedule B (512,384,256,256,192,192; 1024,1024,768,512,384,256): r35 3.224e-8 (+19%) / 8.98e-8 (+4.2%), r100 3.097e-8 (+11%) / 8.03e-8 (+4.4%); schedule C (768,512,384,256,256,192; 1024,1024,1024,768,512,384): r35 2.866e-8 (+6.0%) / 8.607e-8 (-0.2%), r100 2.870e-8 (+3.3%) / 7.538e-8 (-2.1%). The host tolerates C exactly; the +3-6% with the operator on is co-adaptation (refit recoverable). COST when the truncations are billed (Q = the transport map, full rank for 3 ages; randomized SVDs ~1 matrix-matrix multiplication (GEMM)-eq per young cohort; old cohorts ~2.5 vs 4 GEMM-eq): carrier 0.27 -> ~0.20; card ~0.47 with f32 covariance -> adjusted ~1.35e-8 (-24% vs the bank).

Code availability: The shared retrofit instrument retains RETRO_LOWRANK_CP3. The tracker states the age schedules and measurements, but a dedicated run script for these schedules was not identified.

Recorded measurement:

    - T-96 LOW-RANK COHORT BANK, IN-CARD EMULATION (RETRO_LOWRANK_CP3 = "P ranks by age;Q ranks by age": after each push every cohort block is replaced by its rank-truncated SVD, so raw is exactly what a low-rank card gives; independent 4M; deployed r35 2.7038e-8 on / 8.622e-8 off, r100 2.7787e-8 / 7.696e-8): schedule B (512,384,256,256,192,192; 1024,1024,768,512,384,256): r35 3.224e-8 (+19%) / 8.98e-8 (+4.2%), r100 3.097e-8 (+11%) / 8.03e-8 (+4.4%); schedule C (768,512,384,256,256,192; 1024,1024,1024,768,512,384): r35 2.866e-8 (+6.0%) / 8.607e-8 (-0.2%), r100 2.870e-8 (+3.3%) / 7.538e-8 (-2.1%). The host tolerates C exactly; the +3-6% with the operator on is co-adaptation (refit recoverable). COST when the truncations are billed (Q = the transport map, full rank for 3 ages; randomized SVDs ~1 GEMM-eq per young cohort; old cohorts ~2.5 vs 4 GEMM-eq): carrier 0.27 -> ~0.20; card ~0.47 with f32 covariance -> adjusted ~1.35e-8 (-24% vs the bank). A real fire candidate, a day of fnp engineering (factored push/read, randomized SVD, cost certificate), not the 4e-9 class: even with CP4 compressed the same way the floor is ~0.4 (stars 0.14, covariance 0.03, chain/CP4 >= 0.05, misc).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g20_emission_retrofit_v1/run_retro_screen.py](code/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L771](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L771)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-age-dependent-low-rank-source-cohorts-were-emulated-inside) · [Copy provenance](../../../PROVENANCE.json)

