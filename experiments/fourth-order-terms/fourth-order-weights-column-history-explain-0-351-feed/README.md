# Fourth-order weights from column history explain 0.351 of the feed versus 0.60 with freely fitted weights

2026-10-02T09:09Z · experiment · rejected · measured

Whether the free per-column fourth-order weights (fit 0.60 of the feed at layer 13) can be built from how a column evolved was tested with the operators on network 35: rule A at ages 1-2 gives 0.296, rule A at older ages with their own coefficients 0.335, a constant per age group 0.343, and the conversion of a column's own third-order content one layer earlier (diagonal of C, full C or top covariance mode) adds 0.000-0.002; everything together 0.351. The 25 points between 0.35 and 0.60 are not reachable by a per-column rule.

Recorded measurement:

    isotropic 0.212; + rule A ages 1-2 0.296 (coefficient 1.02); + ages 3-4 and 5+ (0.78 and 0.46) 0.335; + constant per age group 0.343; + conversion forms +0.000-0.002; all 0.351; free fit 0.60

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/e34_feed_colweights.py](code/e34_feed_colweights.py)
- [code/physicist/t2_lib.py](code/physicist/t2_lib.py)
- [code/physicist/t4_lib.py](code/physicist/t4_lib.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L223](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L223)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L517](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L517)

<details>
<summary>All result files and records (3)</summary>

- [e34_rank0035_L13.log](e34_rank0035_L13.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-fourth-order-weights-column-history-explain-0-351-feed) · [Copy provenance](../../../PROVENANCE.json)

