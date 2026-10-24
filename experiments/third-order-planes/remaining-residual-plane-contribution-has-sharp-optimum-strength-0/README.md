# Remaining residual-plane contribution has a sharp optimum: strength 0.9 costs 6.15% raw error

2026-09-25T05:43Z · experiment · rejected · measured

After removing the harmful cross term, scaling the remaining off-diagonal star term down caused large losses. The whole-star scale was therefore not added as a dial.

Recorded measurement:

    2026-09-24 22:43 [engineer2] STAR-SCALE PROBE on S21s1 + s1=0 (lead: is the whole star over-represented, or s1 specifically?): star term's off-diagonal sub = s2/3 scaled by c (diagonal unscaled), in-process parity vs s1=0 on 35/100/37/39: c=0.9 MSE +6.15% (per net +10.3/+2.5/+0.9/+12.0%); c=0.8 MSE +21.9% (+25.5/+18.7/+14.3/+30.5%). A quadratic through c=1,0.9,0.8 puts the optimum at c ~1.01 (<= -0.1%). VERDICT: the s2 part sits at its optimum; s1 was the SPECIFIC culprit; no star-scale dial for covhunter. Files: lrroute/starscale_{09,08}_fragment.py, read6/par4_S21s1z_sc{09,08}.log.
    - 2026-09-24 22:43 lead: STAR SCALE CLOSED (engineer2): on S21s1+s1=0, scaling the star's off-diagonal term (s2/3) by c gives c=0.9 +6.15% MSE, c=0.8 +21.9% (4 nets); the quadratic optimum is c ~1.01 (<= -0.1%). s1 was the specific harmful component; the rest of the star sits at its optimum and is sharply curved. No star-scale dial.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/lrroute`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/starscale_08_fragment.py](code/starscale_08_fragment.py)
- [code/starscale_09_fragment.py](code/starscale_09_fragment.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L717](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L717)
- [TEAM_LOG_2026-09-22_to_09-26.md#L718](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L718)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-remaining-residual-plane-contribution-has-sharp-optimum-strength-0) · [Copy provenance](../../../PROVENANCE.json)

