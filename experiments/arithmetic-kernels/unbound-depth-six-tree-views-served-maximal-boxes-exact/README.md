# Unbound depth-six tree views served from maximal boxes: exact warm FLOPs -0.0601%

2026-10-16T07:09:01Z · experiment · inconclusive · measured

Only keys not bound during setup were served from reusable maximal tree boxes, preserving packed leaves for existing keys. This exact view rewrite passed normal and slow gates.

Recorded measurement:

    warm-147.2M/-.0601%;first-164.0M/-.0670%;slowfirst-388.7M/-.158%;requests-8/-12/-28;risk13.32%;setup4.446s,worst109.8s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v1.py](code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v1.py)
- [code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v2.py](code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v2.py)
- [code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v3.py](code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v3.py)
- [code/lead/runs/opus_lazybind_v1/probe_lb_v1.py](code/lead/runs/opus_lazybind_v1/probe_lb_v1.py)
- [code/lead/runs/opus_lazybind_v1/probe_lb_v2.py](code/lead/runs/opus_lazybind_v1/probe_lb_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3987](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3987)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_lazybind_v1.py.txt](fragment_lazybind_v1.py.txt) · result
- [fragment_lazybind_v2.py.txt](fragment_lazybind_v2.py.txt) · result
- [fragment_lazybind_v3.py.txt](fragment_lazybind_v3.py.txt) · result
- [guard_probe3_7103248.json](guard_probe3_7103248.json) · result
- [guard_probe_7103248.json](guard_probe_7103248.json) · result
- [probe3_7103248.json](probe3_7103248.json) · result
- [probe_7103248.json](probe_7103248.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-unbound-depth-six-tree-views-served-maximal-boxes-exact) · [Copy provenance](../../../PROVENANCE.json)

