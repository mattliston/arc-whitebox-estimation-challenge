# Offline rescues of the 224-coordinate read recover only 14 to 39% of the dropped read error (gates half, 75%), closed

2026-10-15T02:03:44Z · finding · rejected · measured

On a captured training network the record priced four ways of recovering the accuracy lost by the 224-coordinate read: an actual-coefficient selector (overlaps 214 of the 224 static coordinates), restoring the exact row mean of the omitted 96, a mean plus rank-2 pivoted completion, and interpolation to 160; each recovered well under the pre-set fraction of the dropped read-error energy, so the 0.5 to 0.6% bill opportunity was closed. An extra B-tree cache capacity audit found the cache already holds every reused tree.

Recorded measurement:

    actual-coefficient selector: 14.194% of summed read-error energy (7.369% Frobenius), overhead 113.059M FLOPs, saving 0.519/0.589%; mean-96 restoration: 37.100% (20.690% Frobenius; per layer 36.08-39.16%), +25.583M FLOPs, opportunity 0.553/0.623%; mean + rank-2 completion: 39.071% (+1.971 pp), +114.321M FLOPs, gate 75%; interpolation 160: 1.116x the zero-160 error, 3.307x the failed 224; adaptive per-layer 224 tariff 0.421/0.491% before basis work; B-cache: UBX 600M -> 1100M saves 0 FLOPs, one extra tree costs 5.181M

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3241](../../TEAM_LOG_2026-10-01_to_10-22.md#L3241)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3258](../../TEAM_LOG_2026-10-01_to_10-22.md#L3258)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3260](../../TEAM_LOG_2026-10-01_to_10-22.md#L3260)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3262](../../TEAM_LOG_2026-10-01_to_10-22.md#L3262)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3264](../../TEAM_LOG_2026-10-01_to_10-22.md#L3264)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3269](../../TEAM_LOG_2026-10-01_to_10-22.md#L3269)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3275](../../TEAM_LOG_2026-10-01_to_10-22.md#L3275)

<details>
<summary>All result files and records (9)</summary>

- [FINAL_OFFLINE_DECISION_v1.json](FINAL_OFFLINE_DECISION_v1.json) · result
- [FIXED_SELECTION_OFFLINE_RESULT_v2.json](FIXED_SELECTION_OFFLINE_RESULT_v2.json) · result
- [FINAL_OFFLINE_DECISION_v1-1d29ba7c.json](FINAL_OFFLINE_DECISION_v1-1d29ba7c.json) · result
- [OFFLINE_MEAN96_RESULT_v1.json](OFFLINE_MEAN96_RESULT_v1.json) · result
- [FINAL_OFFLINE_DECISION_v1-c3479864.json](FINAL_OFFLINE_DECISION_v1-c3479864.json) · result
- [SOURCE_CLOSURE_v1.json](SOURCE_CLOSURE_v1.json) · result
- [SOURCE_ASSESSMENT_v1.json](SOURCE_ASSESSMENT_v1.json) · result
- [CACHE_CAP_SOURCE_DECISION_v1.json](CACHE_CAP_SOURCE_DECISION_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Protected difference read truncated to 224 coordinates: bill -0.61% but MSE +2.54%, adjusted +1.91%, closed](../../../experiments/third-order-planes/protected-difference-read-truncated-224-coordinates-bill-0-61/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-offline-rescues-224-coordinate-read-recover-only-14-39) · [Copy provenance](../../../PROVENANCE.json)

