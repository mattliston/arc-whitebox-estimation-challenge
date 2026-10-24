# Optional execution path causes local footprint growth of 1.15–1.46 GB per network; disabling it stays near 3.2 GB

2026-10-16T18:28:00Z · finding · informational · measured

In one local worker session the server footprint of QL-on builds (the estimator submitted as #336412 (S2tr), V-A, V-A+B) grew by 1.15-1.46 GB per network with no plateau and tripped the 7.5 GB guard at network 5, while the same V-A+B with QL off (compute on) stayed flat over 6 networks. So the growth is the QL path alone and the compute flag is irrelevant; local multi-network guards must use QL-off builds, and the platform (which ran S2tr 5 x 20 networks five times without a memory failure) either does not see the growth or plateaus.

Recorded measurement:

    V-A+B QL on, compute off (15457f15): 3.43 / 4.76 / 6.22 / 7.37 GB; V-A: 3.75 / 5.18 / 6.01 / 7.44, trip at net 5; V-A+B QL off, compute on (9ba7da1a): 3.17 / 3.17 / 3.17 / 3.17 / 3.27 / 3.56 GB over 6 nets.

## Programs

- [run_mn20_v1.sh](run_mn20_v1.sh)
- [cand_vab_qloff_v1.py](cand_vab_qloff_v1.py)
- [cand_vab_compoff_v1.py](cand_vab_compoff_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4446](../../TEAM_LOG_2026-10-01_to_10-22.md#L4446)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4450](../../TEAM_LOG_2026-10-01_to_10-22.md#L4450)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-optional-execution-path-causes-local-footprint-growth-1-151) · [Copy provenance](../../../PROVENANCE.json)

