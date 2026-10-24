# Native covariance-head secants fail all four fixed linearity and curvature cells despite exact baseline replay

2026-09-20 · finding · rejected · measured

Perturbing one covariance-head output coefficient at a time preserved early outputs, local source identities and repeats but failed every complete native response gate. Fresh unchanged-parent replay ruled out stale baseline comparisons; the experiment did not decode targets or establish a full Jacobian.

Recorded measurement:

    Forward/central discrepancies0.704147584,0.576542657,0.933951542,0.457045779;double-step curvatures1.26512434,0.822498317,1.3594502,0.847319223;epsilon1/16;combined15calls/19guards.

## Programs

- [acquire.py](acquire.py)
- [build.py](build.py)
- [common.py](common.py)
- [finish_response.py](finish_response.py)
- [qualify_acquisition.py](qualify_acquisition.py)
- [qualify_response.py](qualify_response.py)
- [run_qualification.py](run_qualification.py)
- [unit.py](unit.py)
- [../../../experiments/covariance-structure/frozen-covariance-edge-correction-passes-panel14-ratio-0-982838708008/code/full_lower_covariance_edge_native_v1/reference.py](../../../experiments/covariance-structure/frozen-covariance-edge-correction-passes-panel14-ratio-0-982838708008/code/full_lower_covariance_edge_native_v1/reference.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)
- [../../../experiments/arithmetic-kernels/caching-four-five-level-left-weight-transforms-saves-1077536896/code/codex_resume_20260919_v1/weight5_reuse_v1/native.py](../../../experiments/arithmetic-kernels/caching-four-five-level-left-weight-transforms-saves-1077536896/code/codex_resume_20260919_v1/weight5_reuse_v1/native.py)

## Evidence

- [RESPONSE_RESULT.md#L12](RESPONSE_RESULT.md#L12)

<details>
<summary>All result files and records (2)</summary>

- [RESPONSE_RESULT.md](RESPONSE_RESULT.md) · record
- [RESULT_after_replay_v1.md](RESULT_after_replay_v1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-native-covariance-head-secants-fail-all-four-fixed-linearity) · [Copy provenance](../../../PROVENANCE.json)

