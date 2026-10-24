# Bounded handle retention can remove six hidden free trips, below the whole-residual target

2026-10-17T07:07:01Z · finding · rejected · measured

Scratch-only retention left eight hidden free trips in eight fully fused depth-6 products because external aliases remained. A four-block batching scheme with a 128 MiB original-owner cap reduced eight trips to two on known owners, but missing complete owner provenance prevented a production fragment.

Recorded measurement:

    Same 593 public calls; 8→2 trips; 24 MiB accounted roots; peak transient handles 74→296; historical 271 list.clear calls; ideal ceiling 203 fewer trips.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4835](../../TEAM_LOG_2026-10-01_to_10-22.md#L4835)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SCRATCH_HOLD_LIFECYCLE_v1.json](SCRATCH_HOLD_LIFECYCLE_v1.json) · result
- [BOUNDED_BH_LIFECYCLE_v1.json](BOUNDED_BH_LIFECYCLE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-bounded-handle-retention-can-remove-six-hidden-free-trips) · [Copy provenance](../../../PROVENANCE.json)

