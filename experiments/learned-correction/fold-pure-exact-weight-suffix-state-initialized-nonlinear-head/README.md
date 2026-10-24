# Fold-pure exact-weight suffix state initialized from the nonlinear head: native baseline still better

2026-08-27 · experiment · rejected · measured

Fold-pure reconstruction and epoch-zero parity pass. C8v2 improves nonlinear-Z by 0.323562% at out-of-fold (OOF) mean squared error (MSE) 3.0439647586e-8, improves all 4 folds, wins 61.63% of networks, and reaches paired z=7.38. It remains scientific-only because native is 2.9736792466e-8, native wins are only 18.90%, and the projected adjusted 1.9422354385e-8 loses to depth-4 third-cumulant arena (D4) #328622.

Recorded measurement:

    Fold-pure reconstruction and epoch-zero parity pass. C8v2 improves nonlinear-Z by 0.323562% at OOF MSE 3.0439647586e-8, improves all 4 folds, wins 61.63% of networks, and reaches paired z=7.38. It remains scientific-only because native is 2.9736792466e-8, native wins are only 18.90%, and the projected adjusted 1.9422354385e-8 loses to D4 #328622.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_q24_c8_nonlinearz_init_v2/run_c8v2_oof.py](code/public_q24_c8_nonlinearz_init_v2/run_c8v2_oof.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_q24_c8_autonomous_correction_state_v1/run_c8_oof.py](code/public_q24_c8_autonomous_correction_state_v1/run_c8_oof.py)
- [code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py](code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py)
- [code/public_q24_nonlinear_transported_raw20_head_v1/run_nonlinear_z.py](code/public_q24_nonlinear_transported_raw20_head_v1/run_nonlinear_z.py)
- [code/public_q24_qhead_full688_refit_v1/run_refit.py](code/public_q24_qhead_full688_refit_v1/run_refit.py)

## Evidence

- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [STATIC_AUDIT.json#L1](STATIC_AUDIT.json#L1)
- [RESULT.json#L1](RESULT.json#L1)
- [RESULT.json#L1](../exact-weight-autonomous-suffix-state-all-four-held-out/RESULT.json#L1)
- [FINAL_FIT_RESULT.json#L1](../exact-weight-autonomous-suffix-state-all-four-held-out/FINAL_FIT_RESULT.json#L1)

<details>
<summary>All result files and records (5)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [STATIC_AUDIT.json](STATIC_AUDIT.json) · result
- [RESULT.json](RESULT.json) · result
- [RESULT.json](../exact-weight-autonomous-suffix-state-all-four-held-out/RESULT.json) · result
- [FINAL_FIT_RESULT.json](../exact-weight-autonomous-suffix-state-all-four-held-out/FINAL_FIT_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-fold-pure-exact-weight-suffix-state-initialized-nonlinear-head) · [Copy provenance](../../../PROVENANCE.json)

