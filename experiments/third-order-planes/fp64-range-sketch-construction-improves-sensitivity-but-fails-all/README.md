# FP64 range-sketch construction improves sensitivity but fails all four response gates at about 10.9% extra cost

2026-09-20 · experiment · rejected · measured

Only shared sketch construction and QR inputs were widened; basis outputs and the remaining estimator returned to float32.

Recorded measurement:

    # FP64 sketch producer: actual native response result
    The single intervention widens only the10 shared range sketches and their QR input.
    Q returns toFP32; core, factors, physical trajectory, selectors and all fitted values otherwise retain original code.
    All7 complete-source inverses pass. The43 tiny/wide FNP helper cases pass against the FP64 reference.
    Fifteen actual cold native calls pass domain/cache checks; base first5 original rows and probe first8 own-base rows are exact.
    The repeated36/head0plus output bytes and full bills match exactly.
    | Rank | Head | Original asymmetry | FP64 sketch asymmetry | Original curvature | FP64 sketch curvature | Pass |
    | 36 | 0 | 0.704147584 | 0.118364379 | 1.26512434 | 0.415537859 | False |
    | 36 | 8 | 0.576542657 | 0.434850387 | 0.822498317 | 0.183713497 | False |
    | 38 | 0 | 0.933951542 | 0.116991237 | 1.3594502 | 0.230165302 | False |
    | 38 | 8 | 0.457045779 | 0.058915474 | 0.847319223 | 0.113551696 | False |
    Each cell requires spacing SNR>=20 and both response ratios<=.10.
    Complete own-cold cost ratios range 1.10880976698..1.10904220279.
    All17 guarded groups retired;15 full native calls;peak sampled6.401GB;zero added swap.
    Artifacts 5.646MB;no new samples/downloads/GPU.
    Official331684 remains3.5844144212297072e-9,100/100 complete. The3/2/~1e-9 milestones remain unmet.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fp64_sketch_response_v1/build.py](code/fp64_sketch_response_v1/build.py)
- [code/fp64_sketch_response_v1/candidate.py](code/fp64_sketch_response_v1/candidate.py)
- [code/fp64_sketch_response_v1/common.py](code/fp64_sketch_response_v1/common.py)
- [code/fp64_sketch_response_v1/finalize.py](code/fp64_sketch_response_v1/finalize.py)
- [code/fp64_sketch_response_v1/qualify.py](code/fp64_sketch_response_v1/qualify.py)
- [code/fp64_sketch_response_v1/run.py](code/fp64_sketch_response_v1/run.py)
- [code/fp64_sketch_response_v1/unit.py](code/fp64_sketch_response_v1/unit.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/shared_sketch_roundoff_diagnostic_v1/snapshot.py](code/shared_sketch_roundoff_diagnostic_v1/snapshot.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L1](RESULT.md#L1)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L5](RESULT.md#L5)
- [REEXAMINATION_INDEX_20260919.md#L773](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L773)
- [REEXAMINATION_INDEX_20260919.md#L777](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L777)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-fp64-range-sketch-construction-improves-sensitivity-but-fails-all) · [Copy provenance](../../../PROVENANCE.json)

