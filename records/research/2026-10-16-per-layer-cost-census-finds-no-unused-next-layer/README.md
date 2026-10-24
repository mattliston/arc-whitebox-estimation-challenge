# Per-layer cost census finds no unused next-layer work in the final layer

2026-10-16T11:51:17Z · finding · rejected · measured

The final layer still required the depth-six push/read leaves for its variance calculation. Profiling did not reveal a removable next-layer-only block.

Recorded measurement:

    layerGF:L0 1.1,L1 5.4,L2 9.6,L3 13.0,L4 16.1,L5 17.9,L6 20.3,L7 21.4,L8 18.4,L9 18.6,L10 21.7,L11 16.9,L12 16.9,L13 19.3,L14 15.8,L15 9.8(4.0%)

## Programs

- [cand_S2tr_layerflops_v1.py](cand_S2tr_layerflops_v1.py)
- [cand_S2tr_layerflops_v2.py](cand_S2tr_layerflops_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4162](../../TEAM_LOG_2026-10-01_to_10-22.md#L4162)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_layerflops_v1.py.txt](fragment_layerflops_v1.py.txt) · result
- [fragment_layerflops_v2.py.txt](fragment_layerflops_v2.py.txt) · result
- [native_7103248.json](native_7103248.json) · result
- [native_v2_7103248.json](native_v2_7103248.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-per-layer-cost-census-finds-no-unused-next-layer) · [Copy provenance](../../../PROVENANCE.json)

