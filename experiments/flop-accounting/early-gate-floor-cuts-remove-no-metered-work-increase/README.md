# Early gate-floor cuts remove no metered work and increase raw error +0.3% to +3.9%

2026-10-19T17:20:48Z · experiment · rejected · measured

Because the error budget places 7% of the final MSE in layers 0-7 and 52% in layers 13-15 while the early pushes and reads are the biggest bills, six cuts were tried in the cheaper direction at the early layers (column cut 1.5 at the reads of layers 1-5, read-tail floor 1.25 at layers 1-6, covariance floor 0.002 at layers 0-6, dead cut 2.5 at the pushes of layers 1-6, young crop 2.0 at layers 7-9, old tail 2.0 at layers 10-11) and all six together. Early units are almost all alive, so the gate floors remove nothing billable while any value change propagates through 10+ layers: the cut-early class is dead on every knob.

Recorded measurement:

    rc_early15 (22 ranks vs stk1): raw +1.077% (se .46, 8/22 lower), bill -0.003%; early_all (22 ranks): raw +3.898% (se .80, 2/22), bill -0.184%; rtt_early125 (28): raw +0.689% (se .28, 9/28), bill -0.000%; dcv_early002: raw +0.022% (se .07), bill +0.070%; dc_early25: raw -0.046%, bill +0.082%; ynt_early20: raw +0.317% (se .22), bill +0.086%; ott_early20: raw +1.790% (se .53, 9/28), bill -0.091%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_rc_early15_sci_v1.py](code/candidate_rc_early15_sci_v1.py)
- [code/candidate_early_all_sci_v1.py](code/candidate_early_all_sci_v1.py)
- [code/candidate_rtt_early125_sci_v1.py](code/candidate_rtt_early125_sci_v1.py)
- [code/candidate_dcv_early002_sci_v1.py](code/candidate_dcv_early002_sci_v1.py)
- [code/candidate_dc_early25_sci_v1.py](code/candidate_dc_early25_sci_v1.py)
- [code/candidate_ynt_early20_sci_v1.py](code/candidate_ynt_early20_sci_v1.py)
- [code/candidate_ott_early20_sci_v1.py](code/candidate_ott_early20_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5975](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5975)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5983](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5983)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5984](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5984)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5986](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5986)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5988](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5988)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5989](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5989)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5992](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5992)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-early-gate-floor-cuts-remove-no-metered-work-increase) · [Copy provenance](../../../PROVENANCE.json)

