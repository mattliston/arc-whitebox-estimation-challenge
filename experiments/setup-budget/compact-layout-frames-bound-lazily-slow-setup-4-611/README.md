# Compact-layout frames bound lazily: slow setup 4.611 to 4.585 seconds for 809,972,800 first-network FLOPs

2026-10-16T01:51:44Z · experiment · adopted · measured

Moving five compact operand frame bindings from setup to the first matching product preserved predictions and removed 95 setup requests. The first network paid the moved view work while warm networks were unchanged.

Recorded measurement:

    setup-95requests;slow4.611to4.585s;normal4.461to4.441s;first+95requests/+809,972,800FLOPs;risk+.44pp

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/cbtrim`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_S1_cblazy_v1.py](code/candidate_S1_cblazy_v1.py)
- [code/candidate_S1_cblazy_v2.py](code/candidate_S1_cblazy_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3660](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3660)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [CB_TRIM_DECISION_v1.json](CB_TRIM_DECISION_v1.json) · result
- [fragment_cb_lazy_v1.py.txt](fragment_cb_lazy_v1.py.txt) · result
- [fragment_cb_lazy_v2.py.txt](fragment_cb_lazy_v2.py.txt) · result

</details>

## Related work

- shipped in: [Submission #336367: layer-10 rank 384, compact kernels and exact request cuts; seven runtime failures](../../../submissions/336367/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-compact-layout-frames-bound-lazily-slow-setup-4-611) · [Copy provenance](../../../PROVENANCE.json)

