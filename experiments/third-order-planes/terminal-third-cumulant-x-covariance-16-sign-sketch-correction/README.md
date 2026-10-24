# Terminal third-cumulant x covariance 16-sign sketch correction: raw ratio 7.19/7.81 on nets 35/100 (adjusted 7.72)

2026-09-20 · experiment · rejected · measured

A terminal readout that contracts one incoming third-cumulant hyperedge with one covariance edge (all retained CP/star/trace owners, inherited final CP pushes, a fixed 16-sign Rademacher first-leg/trace sketch, seed 2026092060) and adds the fourth-order marginal mean response after the unchanged final learned hook passed its arithmetic qualification but multiplied the raw error by about seven on both development networks, failing the requirement of individual ratios below 1.

Recorded measurement:

    raw ratio 7.186844810 (35), 7.805032949 (100); complete cold FLOP ratio 1.030262587 / 1.030007801; adjusted ratio 7.404337329 / 8.039244823, pooled 7.720501701373 vs pooled <=0.995; max FP32 fixture error 1.61524327e-6; seven calls 2,814,212,606,244 F

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_third_covariance_sketch_v1/capture.py](code/terminal_third_covariance_sketch_v1/capture.py)
- [code/terminal_third_covariance_sketch_v1/probes.py](code/terminal_third_covariance_sketch_v1/probes.py)
- [code/terminal_third_covariance_sketch_v1/score.py](code/terminal_third_covariance_sketch_v1/score.py)
- [code/terminal_third_covariance_sketch_v1/instrumented_candidate.py](code/terminal_third_covariance_sketch_v1/instrumented_candidate.py)
- [code/terminal_third_covariance_sketch_v1/instrumented_qualification.py](code/terminal_third_covariance_sketch_v1/instrumented_qualification.py)
- [code/terminal_third_covariance_sketch_v1/unit.py](code/terminal_third_covariance_sketch_v1/unit.py)
- [code/terminal_third_covariance_sketch_v1/finalize.py](code/terminal_third_covariance_sketch_v1/finalize.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/terminal_third_covariance_sketch_v1/common.py](code/terminal_third_covariance_sketch_v1/common.py)
- [code/terminal_third_covariance_sketch_v1/feature_fnp.py](code/terminal_third_covariance_sketch_v1/feature_fnp.py)
- [code/terminal_third_covariance_sketch_v1/reference.py](code/terminal_third_covariance_sketch_v1/reference.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L12](RESULT.md#L12)
- [RESULT.md#L14](RESULT.md#L14)
- [RESULT.md#L17](RESULT.md#L17)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [SOURCE_REVIEW.md](SOURCE_REVIEW.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-terminal-third-cumulant-x-covariance-16-sign-sketch-correction) · [Copy provenance](../../../PROVENANCE.json)

