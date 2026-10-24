# Restoring five or eleven young-read plans saves 0.25% bill but adds about 4% client Python cost

2026-10-16T17:34:47Z · experiment · rejected · measured

The r005 prune had removed 11 Smirnov young-read plans at layer 6 from the clean line. Restoring the 5 with the largest saving per use (S4sm5) passed the request counts but raised Python per request x1.058 on the warm net (median .3323, official 120 s gate FAIL); restoring all 11 (S4sm11) passed the boundary gates (median .3178) but the 6-network session put 5 of 6 networks above .319 and the public-mix bill at only -0.25%, so the #9 fire rule (1.4695e-9) was missed by projection and by residual. The Smirnov read path is Python-heavy per use (about 90M client instructions per restored read use).

Recorded measurement:

    S4sm5 (8b1b7a60): slow 42,115/40,307, worst 94.1 s, Python x1.058 warm, median .3323 FAIL, bill -0.02% vs S4. S4sm11 (4f2e8c6e): slow 42,302/40,531, worst 100.4 s, median .3178 (Python x1.027 first / x1.006 warm), boundary bill -0.215% vs S4, prereg 1.4676e-9; 6-net session: bill mix +0.173% vs S2tr -> 1.4698e-9 (bar 1.4695), residual-instruction ratios 0.919/0.925/0.927/0.960/0.900/0.888 = platform .344 cold, warm .316-.342, median .329.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_passrel_v1/stack_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_s4sm11.py](code/cand_s4sm11.py)
- [code/cand_s4.py](code/cand_s4.py)
- [code/cand_s4sm5.py](code/cand_s4sm5.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4385](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4385)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4387](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4387)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4391](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4391)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4395](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4395)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4397](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4397)

<details>
<summary>All result files and records (6)</summary>

- [fragment_smrestore_top5_v1.py.txt](fragment_smrestore_top5_v1.py.txt) · code
- [fragment_smrestore_top11_v1.py.txt](fragment_smrestore_top11_v1.py.txt) · code
- [r005_plans_v1.json](r005_plans_v1.json) · record
- [STACK_s4sm11.json](STACK_s4sm11.json) · record
- [STACK_s4sm5.json](STACK_s4sm5.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-restoring-five-or-eleven-young-read-plans-saves-0) · [Copy provenance](../../../PROVENANCE.json)

