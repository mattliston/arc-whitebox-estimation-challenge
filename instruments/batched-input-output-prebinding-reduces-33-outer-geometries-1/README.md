# Batched input/output prebinding reduces 33 outer geometries from 1,802 to 600 setup requests

2026-10-17T00:52:55Z · tool · adopted · measured

The batched setup IO fragment extracts the 40 corners of each outer geometry once per side, reuses destination and operand views, then splits publicly into P parts; it preserves values, strides and write-through aliases (tiny RPC proofs on 120 + 192 views) and resets both caches on an arena rebind. It entered the runtime v7+ setup of the peel stack.

Recorded measurement:

    33 outer geometries: 1,802 -> 600 requests (-66.70%), 998,932,480 -> 870,318,080 paid reshape FLOPs, 106 cache entries, warm 0; shared geometries 68 each -> 24/16/16/8 calls.

## Programs

- [code/fragment_batchio_v2.py](code/fragment_batchio_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4600](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4600)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4621](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4621)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-17-batched-input-output-prebinding-reduces-33-outer-geometries-1) · [Copy provenance](../../PROVENANCE.json)

