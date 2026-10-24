# Global context for the exact-weight suffix state: 0.102505% gain missed the 0.25% gate

2026-08-27 · experiment · rejected · measured

The frozen 4x172 out-of-fold (OOF) arm reaches 3.0408445556e-8 versus C8v2 3.0439647586e-8, a 0.102505% gain with all four folds improving, 55.52% network wins, and paired z=3.989. It misses the required 0.25% gain, worsens the protected comparator, loses to same-688 native 2.9736792457e-8, and projects adjusted 1.9402480155e-8 versus depth-4 third-cumulant arena (D4) 1.7836214787e-8.

Recorded measurement:

    The frozen 4x172 OOF arm reaches 3.0408445556e-8 versus C8v2 3.0439647586e-8, a 0.102505% gain with all four folds improving, 55.52% network wins, and paired z=3.989. It misses the required 0.25% gain, worsens the protected comparator, loses to same-688 native 2.9736792457e-8, and projects adjusted 1.9402480155e-8 versus D4 1.7836214787e-8.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_q24_suffix_global_context_v1/run_global_context_oof.py](code/public_q24_suffix_global_context_v1/run_global_context_oof.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_q24_c8_autonomous_correction_state_v1/run_c8_oof.py](code/public_q24_c8_autonomous_correction_state_v1/run_c8_oof.py)
- [code/public_q24_c8_nonlinearz_init_v2/run_c8v2_oof.py](code/public_q24_c8_nonlinearz_init_v2/run_c8v2_oof.py)
- [code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py](code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py)
- [code/public_q24_nonlinear_transported_raw20_head_v1/run_nonlinear_z.py](code/public_q24_nonlinear_transported_raw20_head_v1/run_nonlinear_z.py)
- [code/public_q24_qhead_full688_refit_v1/run_refit.py](code/public_q24_qhead_full688_refit_v1/run_refit.py)

## Evidence

- [RESULT.json#L1](../fold-pure-exact-weight-suffix-state-initialized-nonlinear-head/RESULT.json#L1)
- [RESULT.json#L1](../exact-weight-autonomous-suffix-state-all-four-held-out/RESULT.json#L1)
- [FINAL_FIT_RESULT.json#L1](../exact-weight-autonomous-suffix-state-all-four-held-out/FINAL_FIT_RESULT.json#L1)
- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [STATIC_AUDIT.json#L1](STATIC_AUDIT.json#L1)
- [RESULT.json#L796](RESULT.json#L796)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.json](../fold-pure-exact-weight-suffix-state-initialized-nonlinear-head/RESULT.json) · result
- [RESULT.json](../exact-weight-autonomous-suffix-state-all-four-held-out/RESULT.json) · result
- [FINAL_FIT_RESULT.json](../exact-weight-autonomous-suffix-state-all-four-held-out/FINAL_FIT_RESULT.json) · result
- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [STATIC_AUDIT.json](STATIC_AUDIT.json) · result
- [RESULT.json](RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-global-context-exact-weight-suffix-state-0-102505-gain) · [Copy provenance](../../../PROVENANCE.json)

