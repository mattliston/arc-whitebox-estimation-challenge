# Inherited coefficient scale restored in the basis-selection core increases adjusted error 0.0253%

2026-09-26 · experiment · rejected · measured

The Gram core used for basis selection was rescaled when inherited lambda received its carry multiplier. The coefficient identity passed, but both native screen cases worsened.

Recorded measurement:

    TRAIN2 adjusted+.0253%,both cases worse.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_core_rescale_v1.py](code/candidate_core_rescale_v1.py)
- [code/build_core_rescale_v1.py](code/build_core_rescale_v1.py)
- [code/core_rescale_fragment_v1.py](code/core_rescale_fragment_v1.py)
- [code/candidate_bundle_diagreuse_v1.py](code/candidate_bundle_diagreuse_v1.py)
- [code/native_eval.py](code/native_eval.py)

## Evidence

- [NEXT_RESEARCH.md#L39](../../arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md#L39)

<details>
<summary>All result files and records (5)</summary>

- [NEXT_RESEARCH.md](../../arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md) · record
- [core_rescale_train2_v1_guard.json](core_rescale_train2_v1_guard.json) · result
- [core_rescale_algebra_v1.json](core_rescale_algebra_v1.json) · result
- [core_rescale_manifest_v1.json](core_rescale_manifest_v1.json) · result
- [core_rescale_train2_v1_comparison.json](core_rescale_train2_v1_comparison.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-inherited-coefficient-scale-restored-basis-selection-core-increases-adjusted) · [Copy provenance](../../../PROVENANCE.json)

