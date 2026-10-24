# Bill-ordered prebind blend grows the server to 7.58 GB RSS in setup (8 GB platform cap): depth-7 kernel table unusable

2026-10-19T04:28:02Z · finding · rejected · measured

Weighting the prebind list by billed elements (uses x (1 + elements/3e7)) on the fusion file tripped the 7.5 GB RSS backstop at rank 200 because binding larger trees in setup grows the server process; byte-identical on 35/100 but not fireable under the 8 GB platform cap, so the base kept the count-only refit (same-size swaps). Depth-7 kernel moves are memory-bound too (a 768x4096 C tree at depth 7 is 632 MB of arena), which kills the depth table on this chassis.

Recorded measurement:

    7.58 GB RSS at rank 200 of group 1 (the fusion runs: 7.33 GB); platform cap 8 GB; depth-7 768x4096 C tree = 632 MB arena; full-size bill-ordered prebind: +1,000 requests for -0.37% bill; count x (1 + elements/3e7) blend: -0.03% bill, -14 first-net binds

## Programs

- [candidate_fusion_pb3e7_v1.py](candidate_fusion_pb3e7_v1.py)
- [candidate95c_jsym_pb3e7_v3.py](candidate95c_jsym_pb3e7_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5901](../../TEAM_LOG_2026-10-01_to_10-22.md#L5901)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5904](../../TEAM_LOG_2026-10-01_to_10-22.md#L5904)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-bill-ordered-prebind-blend-grows-server-7-58-gb) · [Copy provenance](../../../PROVENANCE.json)

