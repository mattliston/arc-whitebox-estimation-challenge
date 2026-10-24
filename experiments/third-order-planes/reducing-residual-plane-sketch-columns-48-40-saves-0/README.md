# Reducing residual-plane sketch columns from 48 to 40 saves 0.36% work but costs 0.41% raw error

2026-09-25T02:31Z · experiment · rejected · measured

The smaller range finder worsened raw error on all four screening networks and gave essentially no adjusted benefit.

Recorded measurement:

    - 2026-09-24 19:31 [engineer] Low-rank star sketch 40 vs 48 columns: DEAD (candS18_k40_mem.py vs S18 in process, 37/39/35/100: bill -0.36%, final MSE +0.41% mean, all 4 worse: adjusted about even). S18's k=48 stays; drop item 3 from the small-item bundle.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candS18_k40_mem.py](code/candS18_k40_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L587](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L587)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-reducing-residual-plane-sketch-columns-48-40-saves-0) · [Copy provenance](../../../PROVENANCE.json)

