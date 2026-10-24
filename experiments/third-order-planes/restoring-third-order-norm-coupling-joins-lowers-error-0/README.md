# Restoring third-order norm coupling at joins lowers error by 0.11%

2026-10-02T12:08Z · experiment · rejected · measured

A join projects the joined columns on its rank-r frame and changes their third-order norm-coupling vector by 1.25% (layer 7), 2.30% (layer 10) and 1.02% (layer 13) although it loses only 0.002-0.03% of the columns' energy; the coherent part of the miss can be added back to the transported star for +0.044% of bill. On ten fresh networks the repair moved the final prediction by 0.59-1.14% of the MSE's power but changed the MSE by -0.11% (two standard errors from both a pure removal and an unrelated change). Also measured on two networks: the layer-10 join at rank 384 instead of 320 is break-even (bill +1.70%, MSE -2.79% / -1.56%), ranks 384/256/384 with the repair lose (+2.55% / +5.66%), the old pairs' common part at full rank is no gain.

Recorded measurement:

    norm-coupling change at joins 1.25% / 2.30% / 1.02% vs energy loss 0.03% / 0.03% / 0.002%; repair bill +0.044%, +84 calls; 651 / 378: MSE -1.56% / +0.76%; ten fresh vs S39: MSE -0.11% [-0.74,+0.53] (5 of 10), all-layer -0.28% [-0.70,+0.10], moved power 0.59-1.14% (mean 0.79%); join 384 at layer 10: bill +1.70%, MSE -2.79% / -1.56%; 384/256/384 with repair: bill -1.22%, MSE +2.55% / +5.66%; old pairs' common part full rank layers 10-14: bill +1.33%, MSE -0.60% / -0.52%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a35_join`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_jr_v1.py](code/fragment_jr_v1.py)
- [code/probe_join.py](code/probe_join.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L284](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L284)
- [TEAM_LOG_2026-10-01_to_10-22.md#L289](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L289)
- [TEAM_LOG_2026-10-01_to_10-22.md#L305](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L305)
- [RESULT.md#L1291](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L1291)

<details>
<summary>All result files and records (3)</summary>

- [quick_s39_jr_fresh_a.json](quick_s39_jr_fresh_a.json) · result
- [RESULT.md](../../arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-restoring-third-order-norm-coupling-joins-lowers-error-0) · [Copy provenance](../../../PROVENANCE.json)

