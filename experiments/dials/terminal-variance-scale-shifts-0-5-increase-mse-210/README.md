# Terminal variance scale shifts of 0.5% increase MSE +210..+290%; cumulant scales remain near optimum

2026-10-20T08:51:40Z · experiment · rejected · measured

Four-network native sweeps found terminal variance scaling extremely sensitive and symmetric around the current value. Terminal third/fourth cumulant scalar adjustments offered only noise-level gains.

Recorded measurement:

    [fable-lead] 2026-10-20T08:51:40Z TERMINAL SCALE DIALS AT OPTIMUM (native_cov_oracle_v2.py modes tk3s/tvs/tk4s, nets 35/100/37/39, final MSE vs base): terminal k3 x0.985 +1.02/+0.17/+0.38/+0.89, x1.015 -0.29/+0.45/+0.28/-0.19 (4-net parabola optimum ~x1.007 worth ~-0.1%: noise level); terminal k4 x0.9 +1.9/+2.2/+1.6/-0.3, x1.1 +1.4/+0.9/+1.1/+3.1 (at optimum); terminal v x0.995 / x1.005 = +210..+290% and x0.99 / x1.01 = +840..+1,120% on every net (symmetric: the card's v15 scale is right to <1e-4; a 0.5% v15 error triples the MSE: dm = (phi/2s) dv ~ 1.7e-4 vs the 1.1e-4 error). LAW: the terminal variance is the hyper-sensitive quantity (0.01% accuracy needed); its remaining error (the -10% exact-v15 oracle) is inherited from Cpre14 (-8.7/-15.2 exact oracle), i.e. the late covariance chain. Nothing to fire.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/native_cov_oracle_v2.py](code/native_cov_oracle_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6158](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6158)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-terminal-variance-scale-shifts-0-5-increase-mse-210) · [Copy provenance](../../../PROVENANCE.json)

