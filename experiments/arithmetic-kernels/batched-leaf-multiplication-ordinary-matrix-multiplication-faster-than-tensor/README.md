# Batched leaf multiplication: ordinary matrix multiplication is faster than tensor contraction or outer accumulation

2026-09-26 · experiment · rejected · measured

At depths5,6 and7, ordinary matrix multiplication outperformed einsum and explicit outer-product accumulation. The outer method matched scalar arithmetic counts, while einsum charged slightly more than direct multiplication.

Recorded measurement:

    Depth7 median seconds .0762/.0897/.5501 for matmul/einsum/outer;guard peak2.045GB,zero swap growth.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/leaf_kernel_bench_v1.py](code/leaf_kernel_bench_v1.py)

## Evidence

- [NEXT_RESEARCH.md#L28](NEXT_RESEARCH.md#L28)

<details>
<summary>All result files and records (3)</summary>

- [NEXT_RESEARCH.md](NEXT_RESEARCH.md) · record
- [leaf_kernel_bench_v1.json](leaf_kernel_bench_v1.json) · result
- [leaf_kernel_bench_v1_guard.json](leaf_kernel_bench_v1_guard.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor) · [Copy provenance](../../../PROVENANCE.json)

