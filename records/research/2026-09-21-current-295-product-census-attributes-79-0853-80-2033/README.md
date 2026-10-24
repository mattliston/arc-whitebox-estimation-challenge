# Current 295-product census attributes 79.0853% and 80.2033% of cost to native matrix multiplication

2026-09-21 · finding · informational · derived

Five accounting views independently reconciled complete cold bills and separated outer-product overhead from leaf matrix multiplication.

Recorded measurement:

    # Exact331684 compute frontier: fixed catalog misses standalone3e-9 gate
    Both cold predictions reproduce all16 current-parent output bytes and exact
    complete bills:413448696909F on35 and416264792412F on100. All five accounting
    The current implementation has295 outermost _mm calls per network,
    including two zero-width/zero-bill calls, rather than the older Lazy335.
    All observed operands are FP32. Recursive calls are included exactly once
    |Semantic caller,nonoverlapping charge|Rank35 share|Rank100 share|
    |_push_bank_shared_newborn|20.474%|21.235%|
    |_shared_old384_read|18.114%|17.907%|
    |_slr_reconstruct|13.039%|12.922%|
    |_flms_star|12.206%|12.124%|
    |_shared_old384_push|10.754%|10.696%|
    |_flms_covariance|8.155%|8.099%|
    |_slr_inner|8.112%|7.935%|
    |_slr_join_factors|2.045%|2.029%|
    |gnn_step|1.083%|1.076%|
    |_edge_source|0.778%|0.772%|
    Native matmul-only shares are79.0853% and
    80.2033%. The broader outermost _mm shares
    are92.3119% and
    92.3708%, including transforms,packing
    4.5754%/4.3339%.
    The old reconstruction/inner/rotation group costs about23% of the whole call.
    |Fixed catalog|35 optimistic saving|100 optimistic saving|35 with leaf limit|100 with leaf limit|
    |executable_smirnov_and_bank_winograd|13.546294%|14.413668%|12.658075%|13.501748%|

## Programs

- [build.py](build.py)
- [census.py](census.py)
- [common.py](common.py)
- [finalize.py](finalize.py)
- [prepare.py](prepare.py)
- [../2026-09-21-completing-finite-multiplication-depth-padding-search-adds-zero-saving/reprice.py](../2026-09-21-completing-finite-multiplication-depth-padding-search-adds-zero-saving/reprice.py)
- [run.py](run.py)
- [../2026-09-21-completing-finite-multiplication-depth-padding-search-adds-zero-saving/verify_paths.py](../2026-09-21-completing-finite-multiplication-depth-padding-search-adds-zero-saving/verify_paths.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)
- [census-b7ea375f.py](census-b7ea375f.py)
- [../../../experiments/arithmetic-kernels/finite-multiplication-catalog-delayed-rotation-estimator-saves-most-15/code/reprice.py](../../../experiments/arithmetic-kernels/finite-multiplication-catalog-delayed-rotation-estimator-saves-most-15/code/reprice.py)
- [../../../experiments/arithmetic-kernels/finite-multiplication-catalog-delayed-rotation-estimator-saves-most-15/code/verify_paths.py](../../../experiments/arithmetic-kernels/finite-multiplication-catalog-delayed-rotation-estimator-saves-most-15/code/verify_paths.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [../../../submissions/331599/estimator.py](../../../submissions/331599/estimator.py)
- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L1](RESULT.md#L1)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L10](RESULT.md#L10)
- [REEXAMINATION_INDEX_20260919.md#L720](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L720)
- [REEXAMINATION_INDEX_20260919.md#L723](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L723)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-current-295-product-census-attributes-79-0853-80-2033) · [Copy provenance](../../../PROVENANCE.json)

