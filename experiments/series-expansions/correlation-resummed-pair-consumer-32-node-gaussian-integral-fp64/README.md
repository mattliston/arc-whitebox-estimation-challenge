# Correlation-resummed pair consumer (32-node Gaussian integral, FP64): adjusted 1.0952/1.0985 vs <=0.90, raw 1.001/1.005

2026-09-20 · experiment · rejected · measured

One full native candidate on the unchanged lazy-rotation prefix evaluates the first-order L3+L4 pair map's correlation dependence through conditional Gaussian boundary moments and a fixed 32-node Gauss-Legendre correlation integral (streamed by 8, FP64, |rho|<=0.85), replacing the Gaussian order-6 and cumulant correlation order-2 truncations. Raw error rises slightly and the bill by about 9.4%, so both networks fail the <=0.90 adjusted gate; because the pair arithmetic also moved from the fused FP32 series to FP64, the difference cannot be attributed solely to the omitted correlation terms.

Recorded measurement:

    raw / lazy 1.001070729317 (35), 1.004728245689 (100); own-cold FLOPs / lazy 1.094016528028 / 1.093357030007; adjusted 1.095187923598 / 1.098526690670, pooled 1.096850526741425; enabled cold bills 435,507,103,453 and 438,319,249,228 F; max scaled moment error 9.12072e-15

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/correlation_exact_pair_v1/build.py](code/correlation_exact_pair_v1/build.py)
- [code/correlation_exact_pair_v1/candidate.py](code/correlation_exact_pair_v1/candidate.py)
- [code/correlation_exact_pair_v1/run_native.py](code/correlation_exact_pair_v1/run_native.py)
- [code/correlation_exact_pair_v1/score.py](code/correlation_exact_pair_v1/score.py)
- [code/correlation_exact_pair_v1/finalize.py](code/correlation_exact_pair_v1/finalize.py)
- [code/correlation_exact_pair_v1/unit.py](code/correlation_exact_pair_v1/unit.py)
- [code/correlation_exact_pair_v1/common.py](code/correlation_exact_pair_v1/common.py)
- [code/correlation_exact_pair_v1/kernel.py](code/correlation_exact_pair_v1/kernel.py)
- [code/correlation_exact_pair_v1/null.py](code/correlation_exact_pair_v1/null.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L8](RESULT.md#L8)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L12](RESULT.md#L12)
- [RESULT.md#L21](RESULT.md#L21)
- [RESULT.md#L22](RESULT.md#L22)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [SOURCE_REVIEW.md](SOURCE_REVIEW.md) · record
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-correlation-resummed-pair-consumer-32-node-gaussian-integral-fp64) · [Copy provenance](../../../PROVENANCE.json)

