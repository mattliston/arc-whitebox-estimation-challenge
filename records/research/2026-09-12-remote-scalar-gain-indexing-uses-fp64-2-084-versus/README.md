# Remote scalar gain indexing uses FP64: 2,084 versus 3,090 FLOPs per layer explains the 15,090 FLOPs gap

2026-09-12T21:26Z · finding · informational · derived

A target-free diagnostic replayed the 16 learned gain constants and a deterministic length-1024 vector through both interfaces: native indexed gain arithmetic charged 3,090 F per layer with FP32 tanh/factor/vector, while the remote client's indexed scalar path charged 2,084 F with FP64 values; the 1,006 F per-layer difference over 15 active layers equals exactly the independently observed 15,090 F complete-network discrepancy. A one-element-slice control charged 3,090 F in both interfaces with byte-identical intermediates, which fixed the repair: index the gain as a one-element slice instead of a scalar.

Recorded measurement:

    native 3,090 F vs remote 2,084 F per layer; difference 1,006 F x 15 = 15,090 F; native FP64 hook cast 2,048 F vs remote 0; tiny contexts native 94,780 F vs remote 79,690 F; one-element slice 3,090 F in both

## Evidence

- [SCALAR_SEMANTICS_RESULT_20260912.md#L12](SCALAR_SEMANTICS_RESULT_20260912.md#L12)
- [SCALAR_SEMANTICS_RESULT_20260912.md#L19](SCALAR_SEMANTICS_RESULT_20260912.md#L19)
- [SCALAR_SEMANTICS_RESULT_20260912.md#L49](SCALAR_SEMANTICS_RESULT_20260912.md#L49)

<details>
<summary>All result files and records (2)</summary>

- [SCALAR_SEMANTICS_RESULT_20260912.md](SCALAR_SEMANTICS_RESULT_20260912.md) · record
- [SCALAR_SEMANTICS_PROTOCOL_V3.md](SCALAR_SEMANTICS_PROTOCOL_V3.md) · record

</details>

## Related work

- ← enables: [Matching production OpenBLAS leaves final-layer difference 3.594160079956055e-5 and 15,090 FLOPs bill gap](../../../experiments/hosted-platform/matching-production-openblas-leaves-final-layer-difference-3-594160079956055e/README.md)
- enables: [Corrected scalar gain slicing restores native/client parity and independent14 adjusted score 3.805066974869664e-9](../../../experiments/arithmetic-kernels/corrected-scalar-gain-slicing-restores-native-client-parity-independent14/README.md)

[Research graph](../../../graph/index.html#node=2026-09-12-remote-scalar-gain-indexing-uses-fp64-2-084-versus) · [Copy provenance](../../../PROVENANCE.json)

