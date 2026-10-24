# Local mean-innovation network with 102-32-1 layers has normalized training MSE 0.8766 versus the .50 gate

2026-09-20 · experiment · rejected · measured

A shared FP32 multilayer perceptron (102 inputs: the 30 native features, 64 outgoing hidden values and 8 layer indicators; 3329 scalars) was trained for 100 epochs (2400 updates) to predict the measured local mean innovations of the lazy-rotation card at layers 8 to 15 after subtracting the Gaussian-gain transport of the incoming errors, on TRAIN nets 150/200/250/300/400/500. The terminal normalized TRAIN MSE ratio of 0.8766 against the zero-source predictor fails the <=0.50 gate, so CHECK forecasting and native integration never ran.

Recorded measurement:

    terminal TRAIN normalized MSE / zero 0.8766240975735815 (gate <=0.50; 12.337590% reduction); 100 epochs, 2400 updates, 3329 scalars; Torch output scaled error 3.51146109e-8; parent captures 3,558,435,402,889 F

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lazy_mean_innovation_observer_v1/capture.py](code/lazy_mean_innovation_observer_v1/capture.py)
- [code/lazy_mean_innovation_observer_v1/prepare_data.py](code/lazy_mean_innovation_observer_v1/prepare_data.py)
- [code/lazy_mean_innovation_observer_v1/fit.py](code/lazy_mean_innovation_observer_v1/fit.py)
- [code/lazy_mean_innovation_observer_v1/predict.py](code/lazy_mean_innovation_observer_v1/predict.py)
- [code/lazy_mean_innovation_observer_v1/score.py](code/lazy_mean_innovation_observer_v1/score.py)
- [code/lazy_mean_innovation_observer_v1/unit_fnp.py](code/lazy_mean_innovation_observer_v1/unit_fnp.py)
- [code/lazy_mean_innovation_observer_v1/unit_torch.py](code/lazy_mean_innovation_observer_v1/unit_torch.py)
- [code/lazy_mean_innovation_observer_v1/finalize.py](code/lazy_mean_innovation_observer_v1/finalize.py)
- [code/lazy_covariance_error_transport_v1/common.py](code/lazy_covariance_error_transport_v1/common.py)
- [code/lazy_mean_innovation_observer_v1/common.py](code/lazy_mean_innovation_observer_v1/common.py)
- [code/lazy_mean_innovation_observer_v1/model.py](code/lazy_mean_innovation_observer_v1/model.py)
- [code/lazy_mean_innovation_observer_v1/seal_stage.py](code/lazy_mean_innovation_observer_v1/seal_stage.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L26](RESULT.md#L26)
- [RESULT.md#L27](RESULT.md#L27)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · result
- [fit_result.json](fit_result.json) · result
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-local-mean-innovation-network-102-32-1-layers-has) · [Copy provenance](../../../PROVENANCE.json)

