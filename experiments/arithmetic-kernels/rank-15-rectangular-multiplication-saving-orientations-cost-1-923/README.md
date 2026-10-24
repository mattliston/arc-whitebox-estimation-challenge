# Rank-15 rectangular multiplication: saving orientations cost 1.92–3.25 times CPU

2026-09-26 · experiment · rejected · measured

A metered rank15 rectangular helper was generated from the published2×3×3 coefficient table and verified on every scalar basis pair. Several orientations reduced primitive billed arithmetic, but CPU and request overhead prevented deployment.

Recorded measurement:

    All54 scalar pairs exact;default dispatcher bill+6.6% to+18.9%;depth-six winning shape bill-6.77%,about2xCPU,+1167requests;orientation4 bill-2.64% to-6.77%,CPU1.92–3.25x.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_hk233_v1.py](code/build_hk233_v1.py)
- [code/hk233_fragment_v1.py](code/hk233_fragment_v1.py)
- [code/bench_hk233_v2.py](code/bench_hk233_v2.py)
- [code/bench_hk233_v3.py](code/bench_hk233_v3.py)
- [code/bench_hk233_v1.py](code/bench_hk233_v1.py)
- [code/candidate_bundle_diagreuse_v1.py](code/candidate_bundle_diagreuse_v1.py)

## Evidence

- [NEXT_RESEARCH.md#L45](../batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md#L45)
- [NEXT_RESEARCH.md#L51](../batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md#L51)

<details>
<summary>All result files and records (9)</summary>

- [NEXT_RESEARCH.md](../batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md) · record
- [hk233_bench_v1_guard.json](hk233_bench_v1_guard.json) · result
- [hk233_bench_v2.json](hk233_bench_v2.json) · result
- [hk233_bench_v3.json](hk233_bench_v3.json) · result
- [hk233_bench_v3_guard.json](hk233_bench_v3_guard.json) · result
- [hk233_manifest_v1.json](hk233_manifest_v1.json) · result
- [hk233_bench_v1.json](hk233_bench_v1.json) · result
- [hk233_algebra_v1.json](hk233_algebra_v1.json) · result
- [hk233_bench_v2_guard.json](hk233_bench_v2_guard.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-rank-15-rectangular-multiplication-saving-orientations-cost-1-923) · [Copy provenance](../../../PROVENANCE.json)

