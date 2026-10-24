# Six-network late-term screen: all fourth-order, terminal and variance-floor changes rejected

2026-10-20T04:59:28Z · experiment · rejected · measured

Six-network paired tests rejected the late pair terms, terminal fourth-order scales and covariance-floor changes. Terminal learned-gain and one floor arm were no-ops; late learned gain x0.9 was a small -0.07% MSE effect.

Recorded measurement:

    - [fable-lead] 2026-10-20T04:59:28Z PRESCREEN (native_screen_v2, 6 store nets 35/37/39/100/150/200, paired MSE; the harness's first predict carries +0.25% lazy-bind bill = artifact, fixed by a warm-up): pe34_late_p MSE +0.43%, pe34_late_m +0.06%, pe3_late_p +0.33%, pe4_late_p +0.36%, k415_p +0.31%, k415_m +0.13%, gnng15_p/m and dcvfloor_m no-ops (terminal gain is not P['gain'][15]; floor 0.001 adds no units), dcvfloor_0 MSE -0.15% for bill +0.5%, gnnglate_p +0.09%, gnnglate_m -0.07% (5/6). Dead lanes killed; queue = rel_v4a ref + gnnglate_p/m only.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/native_screen_v2.py](code/native_screen_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6134](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6134)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-six-network-late-term-screen-all-fourth-order-terminal) · [Copy provenance](../../../PROVENANCE.json)

