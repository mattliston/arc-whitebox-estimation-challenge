# CPU estimator mean-encoder refit: first backward pass exceeded the 8 GB process-tree memory guard

2026-09-21T11:02Z · incident · failed · measured

The target-free CPU implementation that adapts the 45,200 learned mean-encoder parameters through the current full estimator passed its translated-operator and FP64 directional-gradient checks and one plain rank-35 forward, but its first layer-checkpointed backward pass was retired by the watchdog at a sampled tree RSS above the 8 GB limit, so the fixed proposal was closed without fitting. A memory audit found 428,081,152 bytes of logically unread old signed-factor prefixes, not enough to explain the overshoot and not a measured RSS saving; this failure motivated the separate allocator-relief qualification.

Recorded measurement:

    sampled tree RSS 8,583,757,824 B vs limit 8,000,000,000 B; plain rank-35 forward 6.099060583 s, process peak 6,793,330,688 B; checkpointed forward 5.716838792 s, peak 7,262,781,440 B; translated operators max relative difference 1.3568390464e-07; six FP64 directional derivatives max discrepancy 9.35932584905e-10; final-layer RMS discrepancy vs native bank 4.08708296561e-06 (all-layer max 2.03847885132e-05)

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L32](RESULT.md#L32)
- [MEMORY_SCOPE.md#L18](MEMORY_SCOPE.md#L18)
- [REEXAMINATION_INDEX_20260919.md#L511](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L511)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](RESULT.md) · result
- [MEMORY_SCOPE.md](MEMORY_SCOPE.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- enables: [Allocator relief at graph boundaries cuts the plain forward high-water from 6,793,330,688 to 2,965,045,248 bytes](../2026-09-21-allocator-relief-graph-boundaries-cuts-plain-forward-high-water/README.md)

[Research graph](../../../graph/index.html#node=2026-09-21-cpu-estimator-mean-encoder-refit-first-backward-pass-exceeded) · [Copy provenance](../../../PROVENANCE.json)

