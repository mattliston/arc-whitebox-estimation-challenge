# Covariance output tiles narrowed to 256 or 128 rows: metered cost rises

2026-09-26 · experiment · rejected · measured

Smaller symmetric output tiles lost fast-multiplication efficiency. The existing 512-row covariance tile was retained.

Recorded measurement:

    A 256-row covariance tile costs +3.32% versus 512 rows; 128 rows costs +20.6%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v1/bench_triangle.py](code/codex_resume_20260926_v1/bench_triangle.py)
- [code/codex_resume_20260926_v1/triangle_fragment_v1.py](code/codex_resume_20260926_v1/triangle_fragment_v1.py)
- [code/fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py](code/fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py)

## Evidence

- [RESULTS.md#L21](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md#L21)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../birth-terms/birth-pair-common-difference-frames-separated-native-adjusted-error/RESULTS.md) · record
- [triangle_bench.json](triangle_bench.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-covariance-output-tiles-narrowed-256-or-128-rows-metered) · [Copy provenance](../../../PROVENANCE.json)

