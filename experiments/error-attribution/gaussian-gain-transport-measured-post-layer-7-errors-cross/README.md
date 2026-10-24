# Gaussian-gain transport of measured post-layer-7 errors: cross-replica final ratio 0.8179 (full) vs the 0.75 gate

2026-09-20 · experiment · rejected · measured

A fixed Gaussian-gain tangent observer initialised with the true sampled post-layer-7 mean and covariance errors of the lazy-rotation card (no later truth resets) forecasts the final error; on net 35 the cross-replica final error falls by about 18.2% (full) against the required 25%, and on net 100 the full/parent ratio is 1.169 against the ordinary 1B reference. One-step resets at every layer give covariance residual cross-ratios 0.32463 (layer 8) to 0.19641 (layer 14) versus 0.84305 for the free forecast, so a single transition is well described but the autonomous forecast loses accuracy without later source information.

Recorded measurement:

    net 35 cross-replica full/parent 0.8179387637, full/mean-only 0.8305631563; net 100 ordinary-1B 1.1694576703 / 0.9877257047; gates both <=0.75; noisy forecast raw ratios 1.1300656207 (35) / 1.1694576703 (100); one-step covariance residual 0.32463 at layer 8, 0.19641 at 14; free forecast 0.84305 at 14, final mean 0.81794; net-35 parent cross error 1.8212163540e-8, observed 8M error 2.7157768645e-8, noise 8.9456051050e-9

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lazy_covariance_error_transport_v1/capture_v2.py](code/lazy_covariance_error_transport_v1/capture_v2.py)
- [code/lazy_covariance_error_transport_v1/analyze.py](code/lazy_covariance_error_transport_v1/analyze.py)
- [code/lazy_covariance_error_transport_v1/finalize.py](code/lazy_covariance_error_transport_v1/finalize.py)
- [code/lazy_covariance_error_transport_v1/unit_v2.py](code/lazy_covariance_error_transport_v1/unit_v2.py)
- [code/lazy_covariance_error_transport_v1/common.py](code/lazy_covariance_error_transport_v1/common.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L10](RESULT.md#L10)
- [RESULT.md#L12](RESULT.md#L12)
- [RESULT.md#L22](RESULT.md#L22)
- [RESULT.md#L23](RESULT.md#L23)
- [RESULT.md#L25](RESULT.md#L25)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [DEPENDENCY_FIX.md](DEPENDENCY_FIX.md) · record
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-gaussian-gain-transport-measured-post-layer-7-errors-cross) · [Copy provenance](../../../PROVENANCE.json)

