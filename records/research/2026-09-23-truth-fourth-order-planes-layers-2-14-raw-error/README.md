# Truth fourth-order planes at layers 2-14: raw error falls 12.2% and 14.7% on two networks

2026-09-23T02:10Z · finding · informational · measured

Replacing the (3,1) and (2,2) planes by Monte Carlo truth improved raw error through the covariance output; the (2,1) off-diagonal substitution worsened it.

Recorded measurement:

    2026-09-22 19:10 physicist: ORACLE (torch, no refit, pair map only, L2-14): truth K31 alone r35 -12.1%; K22+K31 r35 -12.2% r100 -14.7%; +truth k4 r35 -21.4% r100 -8.7%; truth K21 off-diag +430%/+827%. K31 error top SVD pair = k3 (x) mu (gain-on-skewed-base term). fable_physicist_v1/oracle_planes.jsonl
    2026-09-22 19:14 physicist: (timestamp correction: the two 19:10 lines above were written ~18:55.) Truth-K31 oracle gain is signal (independent-draw planes at L2/5/9/12: -5.8% vs main -4.7%), enters ONLY via the pair map's covariance output -> next-layer propagation; broad-spectrum (top-50 SVD needed). Zero-cost per-row K31 = rad v_a C_ac (0.85 + 0.5 mu_a k3_a/v_a^2): -1.27/-1.61/-0.19/-3.03% on 35/100/37/39 (torch, no refit).

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L9](../../TEAM_LOG_2026-09-22_to_09-26.md#L9)
- [TEAM_LOG_2026-09-22_to_09-26.md#L15](../../TEAM_LOG_2026-09-22_to_09-26.md#L15)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [oracle_planes.jsonl](oracle_planes.jsonl) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-truth-fourth-order-planes-layers-2-14-raw-error) · [Copy provenance](../../../PROVENANCE.json)

