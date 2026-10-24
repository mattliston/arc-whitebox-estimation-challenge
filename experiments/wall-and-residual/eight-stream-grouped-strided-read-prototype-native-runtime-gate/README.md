# Eight-stream grouped strided-read prototype: native runtime gate failed

2026-09-06T07:54Z · experiment · rejected · measured

All six Smirnov and36 catalog executable rational circuits passed their exact identities, including explicit packed-copy accounting. The final same-census border-copy envelope saved20.0598/20.2591% optimistically on rank35/100. One-thread warm: .932035s versus bankD4 .115650s, ratio8.05911.

Recorded measurement:

    All six Smirnov and36 catalog executable rational circuits passed their exact identities, including explicit packed-copy accounting. The final same-census border-copy envelope saved20.0598/20.2591% optimistically on rank35/100. This passed the separately declared20% feasibility gate; it did not retroactively pass the older25% research gate. Fixed native FNP0.12 product1024x832 times832x8192: three primary Smirnov stages, top-rank groups8, known streamed Winograd leaf and exact inner fringe. Retained workspace1,094,241,296 bytes (1GiB arena plus20,499,472 dedicated padding); peak live arena595,483,520. Work-memory cap PASSES. Small iid certificate relative error1.17602e-4. Full relative Frobenius error1.44725e-4, finite, within the declared2e-4 bound, but21.7x the bank's 6.66779e-6. Actual structured CP values were NOT tested after runtime failed. One-thread warm: .932035s versus bankD4 .115650s, ratio8.05911. Four-thread warm: .940316s versus .109023s, ratio8.62491. Both fail the3x gate. Warm bill6,611,249,012 versus8,417,538,304 FLOPs, a21.4586% PRODUCT-only saving; cold prototype6,834,384,092. This is NOT a whole-card20% measured saving and NOT a native-cost match to the earlier arithmetic envelope. Complete process peak RSS1,795,981,312 bytes includes both kernel workspaces and reference matrices; not a grader memory bound.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/benchmark_smirnov_stream8_v1.py](code/benchmark_smirnov_stream8_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/packed_arena_generic.py](code/packed_arena_generic.py)
- [code/smirnov_stream8_native_v1.py](code/smirnov_stream8_native_v1.py)
- [code/streamed_leaf_winograd_v1.py](code/streamed_leaf_winograd_v1.py)

## Evidence

- [SMIRNOV_STREAM8_PROTOTYPE_V1_CLOSED.md#L1](SMIRNOV_STREAM8_PROTOTYPE_V1_CLOSED.md#L1)
- [SMIRNOV_STREAM8_PROTOTYPE_V1_CLOSED.md#L8](SMIRNOV_STREAM8_PROTOTYPE_V1_CLOSED.md#L8)
- [SMIRNOV_STREAM8_PROTOTYPE_V1_CLOSED.md#L19](SMIRNOV_STREAM8_PROTOTYPE_V1_CLOSED.md#L19)

<details>
<summary>All result files and records (3)</summary>

- [SMIRNOV_STREAM8_PROTOTYPE_V1_CLOSED.md](SMIRNOV_STREAM8_PROTOTYPE_V1_CLOSED.md) · record
- [smirnov_stream8_native_benchmark_v1.json](smirnov_stream8_native_benchmark_v1.json) · result
- [SMIRNOV_STREAM8_PROTOTYPE_DESIGN_V1.md](SMIRNOV_STREAM8_PROTOTYPE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-eight-stream-grouped-strided-read-prototype-native-runtime-gate) · [Copy provenance](../../../PROVENANCE.json)

