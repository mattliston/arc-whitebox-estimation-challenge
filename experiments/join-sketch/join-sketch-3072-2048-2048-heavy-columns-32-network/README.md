# Join sketch on 3072/2048/2048 heavy columns: 32-network adjusted -1.431% with unchanged raw error

2026-10-18T01:27:26Z · experiment · informational · measured

Only the heaviest joined columns formed the Nyström sketch Gram at layers 7, 10 and 13, while factors remained exact over all columns and the core was rescaled to full trace. This single-product form preserved noise robustness and confirmed a cost-side saving without a power step.

Recorded measurement:

    32 ranks: adjusted -1.431% (SE .102), raw -0.012% (SE .093, 16/32 lower), bill -1.419%; versus all-shape union on 25 ranks: adjusted -1.248%, bill -1.257%; rescue activated on 0/33.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_v52j.py](code/candidate_v52j.py)
- [code/jsk/make_joinsketch_v6.py](code/jsk/make_joinsketch_v6.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5033](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5033)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5034](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5034)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5037](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5037)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5039](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5039)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5050](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5050)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [basis35_v3.txt](basis35_v3.txt) · result
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result
- [GRADE_STORE_v1-205c20f5.json](GRADE_STORE_v1-205c20f5.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-join-sketch-3072-2048-2048-heavy-columns-32-network) · [Copy provenance](../../../PROVENANCE.json)

