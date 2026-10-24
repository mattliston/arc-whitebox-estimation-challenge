# Buffered iteration of short strided views consumes one third of transform time

2026-10-02T07:39Z · finding · informational · measured

On small numpy probes, an elementwise operation on a strided view runs at 1.3-2.6 ns per float instead of 0.3 when the contiguous run is under 4,096 floats; in the tree kernel that is levels 1 and 2 at 16 x 16 leaves (runs of 256 and 1,792 floats), 7% of a transform's elements and about a third of its time. Reordering the level axes cannot remove it; leaf size and depth decide how many levels sit under the threshold.

Recorded measurement:

    1.3-2.6 ns per float (runs < 4,096) vs 0.3; levels 1-2 at 16 x 16 leaves: runs 256 and 1,792 floats, 7% of elements, about a third of the time

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L181](../../TEAM_LOG_2026-10-01_to_10-22.md#L181)
- [RESULT.md#L1096](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L1096)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-buffered-iteration-short-strided-views-consumes-one-third-transform) · [Copy provenance](../../../PROVENANCE.json)

