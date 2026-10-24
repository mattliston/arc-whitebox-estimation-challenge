# Acceptance rule uses 16 research screens, eight fresh native checks and negative mean adjusted change

2026-10-12T17:19:17Z · decision · informational · measured

From the analyst's podium curve (P(podium) 17.3% at our score x1.000, 30.5% at 0.990, 48.9% at 0.980, 72.4% at 0.970: about 13 points per 1% with a second difference of about 3.2 points per %^2) radlab derived and the lead adopted the rule for scout and structural levers: (1) a paired PyTorch screen on 16 networks (stop at 4 if every arm loses by > 3 se), closing a lever if its PyTorch error rise alone beats the metered FLOPs saving at 95% because error-adding changes amplify natively; (2) a pre-registered native check on 8 fresh networks paired against the current file giving the score change S; (3) submission by podium value, S_mean < 0.12 x sd^2, in practice submission iff the 8-network native mean < 0 and extend to 16 between 0 and +1 se; (4) release's residual / wall / request gates unchanged; small refit rounds keep their own rules.

Recorded measurement:

    podium curve 17.3% / 30.5% / 48.9% / 72.4% at x1.000 / 0.990 / 0.980 / 0.970; about 13 points per 1%, second difference about 3.2 points per %^2; fire iff S_mean < 0.12 x sd^2

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1567](../../TEAM_LOG_2026-10-01_to_10-22.md#L1567)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1570](../../TEAM_LOG_2026-10-01_to_10-22.md#L1570)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-acceptance-rule-uses-16-research-screens-eight-fresh-native) · [Copy provenance](../../../PROVENANCE.json)

