# Exact carried covariance reduces raw error 29.6-35.3%; fitted carrier weights instead worsen it 2.2-16x

2026-09-23T04:27Z · finding · informational · measured

Truth substitutions show substantial value in late carried covariance. Least-squares reweighting of the existing third-order carriers is unstable in the deployed recurrence.

Recorded measurement:

    2026-09-22 21:27 covhunter: per-layer VALUE of the exact carried covariance (torch, learned ON, no refit; C after step l -> 32M truth, single layer), final-MSE change r35 / r100: L1 +3.3/+0.5, L2 +0.6/+0.6, L3 0.0/+0.3, L4 -4.0/-0.7, L5 -6.6/-4.4, L6 -10.2/-5.8, L7 -13.1/-5.3, L8 -12.7/-12.2, L9 -9.1/-16.0, L10 -8.2/-20.8, L11 -10.5/-16.6, L12 -12.7/-17.5, L13 -9.6/-13.8, L14 -5.6/-11.1; ALL layers -29.6/-35.3; all layers + one extra 32M noise draw -22.0/-26.2 (the truth's own noise costs ~8 pts; L1-3 at the noise floor). Carrier reweighting of the kink-row plane in the covariance call (least-squares a*atoms+b*star+c*bterms vs truth: coefficients universal across nets, a .9->.55, b 2.3->5, c 1.5->2.5 over L4-14, residual 45%->33%): CATASTROPHIC closed loop (x2.2-16), because the atoms read is a stiff direction (adjoint price per doubling -36..+117% per layer, sign alternating per layer and net). swap_v1.jsonl, clcarrier_v1.jsonl, k21_decomp_rank*.json.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L48](../../TEAM_LOG_2026-09-22_to_09-26.md#L48)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [swap_v1.jsonl](swap_v1.jsonl) · result
- [clcarrier_v1.jsonl](clcarrier_v1.jsonl) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-exact-carried-covariance-reduces-raw-error-29-6-35) · [Copy provenance](../../../PROVENANCE.json)

