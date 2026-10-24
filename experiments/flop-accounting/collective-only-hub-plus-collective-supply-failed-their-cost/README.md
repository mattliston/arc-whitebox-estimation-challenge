# Collective-only and hub-plus-collective supply failed their cost-adjusted submission thresholds

2026-08-31 · experiment · rejected · measured

GATE-1 VERDICT (5-net leave-one-out (LOO), deploy-native calibrations): collective-only supply (_gc, zero bill) -1.3% +- 1.2 - FAILS the 3% bar; joint hub+collective (_gj) +2.5% +- 3.6 - fails both its ~27% bill-adjusted bar and the 3% floor. NO FIRE. The supply-quality -> capture curve is now measured: q 0.38 -> -1.3%, q 0.62 -> +2.5%, q 0.78 -> +15.8% - steeply superlinear near the top; the conversion threshold behavior is confirmed panel-grade.

Recorded measurement:

    GATE-1 VERDICT (5-net LOO, deploy-native calibrations): collective-only supply (_gc, zero bill) -1.3% +- 1.2 - FAILS the 3% bar; joint hub+collective (_gj) +2.5% +- 3.6 - fails both its ~27% bill-adjusted bar and the 3% floor. NO FIRE. The supply-quality -> capture curve is now measured: q 0.38 -> -1.3%, q 0.62 -> +2.5%, q 0.78 -> +15.8% - steeply superlinear near the top; the conversion threshold behavior is confirmed panel-grade. Fire remains blocked on P2R0G19B_CARRIER.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_loo_grade.py](code/run_loo_grade.py)

## Evidence

- [loo_grade_gc.log#L1](loo_grade_gc.log#L1)
- [loo_grade_gj.log#L1](loo_grade_gj.log#L1)

<details>
<summary>All result files and records (2)</summary>

- [loo_grade_gc.log](loo_grade_gc.log) · log
- [loo_grade_gj.log](loo_grade_gj.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-31-collective-only-hub-plus-collective-supply-failed-their-cost) · [Copy provenance](../../../PROVENANCE.json)

