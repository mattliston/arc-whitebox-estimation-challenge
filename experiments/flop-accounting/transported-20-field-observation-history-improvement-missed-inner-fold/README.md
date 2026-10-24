# Transported 20-field observation history: improvement missed the inner-fold gate

2026-08-27 · experiment · rejected · measured

Keeping all 20 standardized native fields as separate Phi*W states does not clear the inner gate. The best layerwise-delta-q ridge arm moves inner mean squared error (MSE) from 3.158339528e-8 to 3.151125611e-8, ratio 0.9977159 versus the required 0.99, while adding 546,068,480 FLOPs. Protected40 remained unopened.

Recorded measurement:

    Keeping all 20 standardized native fields as separate Phi*W states does not clear the inner gate. The best layerwise-delta-q ridge arm moves inner MSE from 3.158339528e-8 to 3.151125611e-8, ratio 0.9977159 versus the required 0.99, while adding 546,068,480 FLOPs. Protected40 remained unopened.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py](code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py)
- [code/public_q24_native_j_transported_raw20_bank_v1/run_screen.py](code/public_q24_native_j_transported_raw20_bank_v1/run_screen.py)

## Evidence

- [RESULT.json#L149](../../learned-correction/endpoint-action-correction-head-local-projection-failed-transfer-public/RESULT.json#L149)

<details>
<summary>All result files and records (1)</summary>

- [RESULT.json](../../learned-correction/endpoint-action-correction-head-local-projection-failed-transfer-public/RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-transported-20-field-observation-history-improvement-missed-inner-fold) · [Copy provenance](../../../PROVENANCE.json)

