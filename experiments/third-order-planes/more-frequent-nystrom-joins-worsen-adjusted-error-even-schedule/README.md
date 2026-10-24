# More frequent Nyström joins worsen adjusted error: even schedule about +0.3%

2026-09-25T07:16Z · experiment · rejected · measured

Rechecking the schedule after cheaper joins found that extra truncations still hurt accuracy. Their native work saving was too small to compensate.

Recorded measurement:

    2026-09-25 00:16 covhunter: JOIN SCHEDULE RE-CHECK (Nystrom, S23 card, loopR1 + round-1 dials, no refit; RESULT sec 31): VALID16 raw vs every3: even +0.57% [CI +0.04,+1.05] 4/16; four7 +1.18% [+0.51,+1.88] 3/16. More joins still cost accuracy. Adjusted bar (-0.8%) needs native bill <= -2.2% (even) / -2.8% (four7) -> likely KILL; porter's native bills decide.
    2026-09-25 00:18 covhunter: JOIN SCHEDULE RE-CHECK KILLED (RESULT sec 31): porter's native bill even vs every3 -0.07% (r35) / -0.47% (r100) ~ -0.3%, against VALID16 raw +0.57% -> adjusted ~+0.3% (bar -0.8%). four7 (raw +1.18%) not built. every3 stays.
    - 2026-09-25 00:18 lead: JOIN SCHEDULE RE-CHECK KILLED (covhunter torch + porter native): even +0.57% raw at bill ~-0.3% = adjusted ~+0.3%; four7 +1.18% raw. every3 stays; the schedule is set by accuracy (each join re-truncates the old tier). Covhunter next: learned-part CAPACITY test on MPS (control vs function-preserving wide2x, 2 epochs each from loopR1, S23-structure card; bar: wide beats control by >= 0.5% VALID16), queued behind the physicist's loopR3.
    2026-09-25 00:19 porter: JOIN SCHEDULE line KILLED (covhunter + lead): even vs every3 torch VALID16 raw +0.57% [+0.04, +1.05] 4/16 against my native bill -0.07%/-0.47% (r35/r100) = adjusted ~+0.3% (bar -0.8%); four7 {7:(0,2),9:(3,4),11:(5,6),13:(7,8)} raw +1.18% -> NOT built. every3 stays. Porter standing by for the next real change (E-71 group 1 / loopR3 / depth-routing), to be composed with star-Nystrom (stack/lrnys_fragment_q1.py) + residual gate on the composition.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1/stack`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lrnys_fragment_q1.py](code/lrnys_fragment_q1.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L818](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L818)
- [TEAM_LOG_2026-09-22_to_09-26.md#L819](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L819)
- [TEAM_LOG_2026-09-22_to_09-26.md#L820](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L820)
- [TEAM_LOG_2026-09-22_to_09-26.md#L823](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L823)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-more-frequent-nystrom-joins-worsen-adjusted-error-even-schedule) · [Copy provenance](../../../PROVENANCE.json)

