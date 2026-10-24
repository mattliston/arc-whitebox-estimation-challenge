# Rank-23 three-by-three multiplication loses 4.66–19.90% compute and costs 2.5–3.7 times CPU

2026-09-26 · experiment · rejected · measured

The restricted parser generated a metered helper from the published3×3×3 rank23 multiplication table. All scalar basis products and block fixtures passed, but all four native-shape screens were more expensive.

Recorded measurement:

    All81 scalar basis products exact;bill+4.66% to+19.90%,CPU2.5–3.7x across4 shapes;no estimator candidate.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_laderman333_v1.py](code/build_laderman333_v1.py)
- [code/laderman333_fragment_v1.py](code/laderman333_fragment_v1.py)
- [code/bench_laderman333_v1.py](code/bench_laderman333_v1.py)
- [code/candidate_bundle_diagreuse_v1.py](code/candidate_bundle_diagreuse_v1.py)

## Evidence

- [NEXT_RESEARCH.md#L49](../batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md#L49)

<details>
<summary>All result files and records (5)</summary>

- [NEXT_RESEARCH.md](../batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md) · record
- [laderman333_algebra_v1.json](laderman333_algebra_v1.json) · result
- [laderman333_bench_v1.json](laderman333_bench_v1.json) · result
- [laderman333_manifest_v1.json](laderman333_manifest_v1.json) · result
- [laderman333_bench_v1_guard.json](laderman333_bench_v1_guard.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-rank-23-three-three-multiplication-loses-4-6619-90) · [Copy provenance](../../../PROVENANCE.json)

