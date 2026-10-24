# Batched same-key tree products: only pairs of equal chunks fit the arena, about 80 requests per network, not built

2026-10-16T17:51:31Z · finding · rejected · measured

A feasibility census on S4 asked whether the chunks of a k-split product read could share one forward transform pass. Batching all chunks would save 962 cold / 1,256 warm requests, but full-size chunk trees fill their depth-6 arena regions, byte-identity needs equal chunk sizes in a pair, and A-side pairs never fit their region, so the realistic yield was 4 pairs of N = 384 (about 80 requests per network) without a custom whole-arena layout. Stopped as a 4-hour kernel change worth about -0.1% bill via two-strip reverts.

Recorded measurement:

    Per network 30-31 k-split reads, 1-4 chunks each, forward-transform requests 20-59 per chunk pass, k-split reads = 4,463 warm requests; all chunks batched -962 cold / -1,256 warm; pairs only 728 cold / 862 warm; equal adjacent pairs 14 warm (~648) / 22 cold (~776); B pairs fit only for N = 384 (4 pairs, ~80 requests/net); largest pairs overflow ([992, 6144, 928] needs 406.6M of the 391.6M-float arena).

## Programs

- [build_btprobe_v2.py](build_btprobe_v2.py)
- [cand_s4_btprobe_v2.py](cand_s4_btprobe_v2.py)
- [../../../experiments/wall-and-residual/restoring-five-or-eleven-young-read-plans-saves-0/code/cand_s4.py](../../../experiments/wall-and-residual/restoring-five-or-eleven-young-read-plans-saves-0/code/cand_s4.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4409](../../TEAM_LOG_2026-10-01_to_10-22.md#L4409)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4412](../../TEAM_LOG_2026-10-01_to_10-22.md#L4412)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4413](../../TEAM_LOG_2026-10-01_to_10-22.md#L4413)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4419](../../TEAM_LOG_2026-10-01_to_10-22.md#L4419)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4421](../../TEAM_LOG_2026-10-01_to_10-22.md#L4421)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-batched-same-key-tree-products-only-pairs-equal-chunks) · [Copy provenance](../../../PROVENANCE.json)

