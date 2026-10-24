# Boxing every depth-six tree key saves slightly more FLOPs but raises server instructions 13-15%

2026-10-16T07:15:37Z · experiment · rejected · measured

Extending boxed views to already-prebound keys made the leaf products strided and slower. The restricted unbound-key version was retained.

Recorded measurement:

    warm-.0642%,first-.0810%;requests-20/-62;serverinstructions+13..15%,cycles+18..38%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v1.py](code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v1.py)
- [code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v2.py](code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v2.py)
- [code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v3.py](code/lead/runs/opus_lazybind_v1/cand_GOV3_lazybind_v3.py)
- [code/lead/runs/opus_lazybind_v1/probe_lb_v1.py](code/lead/runs/opus_lazybind_v1/probe_lb_v1.py)
- [code/lead/runs/opus_lazybind_v1/probe_lb_v2.py](code/lead/runs/opus_lazybind_v1/probe_lb_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3988](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3988)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_lazybind_v1.py.txt](../unbound-depth-six-tree-views-served-maximal-boxes-exact/fragment_lazybind_v1.py.txt) · result
- [fragment_lazybind_v2.py.txt](../unbound-depth-six-tree-views-served-maximal-boxes-exact/fragment_lazybind_v2.py.txt) · result
- [fragment_lazybind_v3.py.txt](../unbound-depth-six-tree-views-served-maximal-boxes-exact/fragment_lazybind_v3.py.txt) · result
- [guard_probe3_7103248.json](../unbound-depth-six-tree-views-served-maximal-boxes-exact/guard_probe3_7103248.json) · result
- [guard_probe_7103248.json](../unbound-depth-six-tree-views-served-maximal-boxes-exact/guard_probe_7103248.json) · result
- [probe3_7103248.json](../unbound-depth-six-tree-views-served-maximal-boxes-exact/probe3_7103248.json) · result
- [probe_7103248.json](../unbound-depth-six-tree-views-served-maximal-boxes-exact/probe_7103248.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-boxing-every-depth-six-tree-key-saves-slightly-more) · [Copy provenance](../../../PROVENANCE.json)

