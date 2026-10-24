# Endpoint fine-tuning of the same correction-head features: improvement missed the inner-fold gate

2026-08-27 · experiment · rejected · measured

Autonomous endpoint fine-tuning of the same 6,022 q-head parameters selects epoch4 at inner mean squared error (MSE) 3.154637973e-8, only ratio 0.9988280 against the frozen head's 3.158339528e-8 and short of the required 0.99. It adds no inference FLOPs, but protected40 remained unopened because the information-limited gate failed.

Recorded measurement:

    Autonomous endpoint fine-tuning of the same 6,022 q-head parameters selects epoch4 at inner MSE 3.154637973e-8, only ratio 0.9988280 against the frozen head's 3.158339528e-8 and short of the required 0.99. It adds no inference FLOPs, but protected40 remained unopened because the information-limited gate failed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py](code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py)
- [code/public_q24_native_j_qhead_action_finetune_v1/run_discriminator.py](code/public_q24_native_j_qhead_action_finetune_v1/run_discriminator.py)

## Evidence

- [RESULT.json#L25](RESULT.json#L25)

<details>
<summary>All result files and records (1)</summary>

- [RESULT.json](RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-endpoint-fine-tuning-same-correction-head-features-improvement-missed) · [Copy provenance](../../../PROVENANCE.json)

