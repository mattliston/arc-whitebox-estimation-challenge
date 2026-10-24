# Garbage-collector freezing leaves billed residual instructions unchanged

2026-10-16T05:34:45Z · experiment · rejected · measured

Callback profiling found sizable collection pauses, but they mostly fell within dispatch overhead rather than the participant residual meter. Freezing did not provide the required residual reduction.

Recorded measurement:

    gen0~25-34ms,180-235collections;gen1~6ms;gen20-29ms,max28.7ms;freezogen22.5-13ms;instructions45.30-45.61Kvs45.34-45.55K

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_gc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_S2tr_gcdiag_v1.py](code/cand_S2tr_gcdiag_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3914](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3914)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_gc_diag_v1.py.txt](fragment_gc_diag_v1.py.txt) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-garbage-collector-freezing-leaves-billed-residual-instructions-unchanged) · [Copy provenance](../../../PROVENANCE.json)

