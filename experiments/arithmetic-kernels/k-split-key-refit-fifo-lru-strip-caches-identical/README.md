# K-split key refit and FIFO-to-LRU strip caches: identical predictions, 0.03% fewer FLOPs

2026-10-16T04:26:26Z · experiment · adopted · measured

Only the main prebind list was materially stale; a smaller accumulator list and strip-cache eviction policy supplied minor exact savings. Worker-position residual analysis did not support cache churn as the platform failure driver.

Recorded measurement:

    26/40accumulatorkeysunused,24unlisted;treeusage254/321/260vscaps320/352/320;first-16/warm-8;FLOPs-.03%;risk14.86to14.78%;residualpositioncorr-.03..-.12

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_tables_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_S4_hzlru_v1.py](code/candidate_S4_hzlru_v1.py)
- [code/candidate_S4_tables_v1.py](code/candidate_S4_tables_v1.py)
- [code/cache_probe_v1.py](code/cache_probe_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3858](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3858)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3860](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3860)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3863](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3863)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_hz_lru_v1.py.txt](fragment_hz_lru_v1.py.txt) · result
- [fragment_ks_acc_refit_v1.py.txt](fragment_ks_acc_refit_v1.py.txt) · result
- [TABLES_AUDIT_v1.json](TABLES_AUDIT_v1.json) · result
- [cache_probe_S4_v1.json](cache_probe_S4_v1.json) · result
- [guard_probe_v1.json](guard_probe_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336412: rank-384 stack with own prebind census, accumulator refit and least-recently-used caches](../../../submissions/336412/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-k-split-key-refit-fifo-lru-strip-caches-identical) · [Copy provenance](../../../PROVENANCE.json)

