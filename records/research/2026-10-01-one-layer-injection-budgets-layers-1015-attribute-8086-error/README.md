# One-layer injection budgets at layers 10–15 attribute 80–86% of error to inherited inputs

2026-10-01T19:20Z · finding · informational · measured

On the torch card (networks 35, 100, 37, 39 pooled) each per-unit input of the mean formula was replaced by its truth one layer at a time. Variance injects 1.5-2.0, skew 0.8-1.2 and kurtosis 1.0-1.3 (x1e-9) per layer at layers 10-15, 80-86% of the previous layer's error is carried on, the learned correction removes 3-5% per layer, and the mean formula with truth inputs errs below 0.1.

Recorded measurement:

    per layer x1e-9: variance 1.5-2.0, skew 0.8-1.2, kurtosis 1.0-1.3, combined 2.7-3.8; carried-on share 80-86% (not 75%); learned correction -3..-5% per layer; mean formula with truth inputs < 0.1

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L11](../../TEAM_LOG_2026-10-01_to_10-22.md#L11)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../2026-10-01-torch-research-estimator-lowers-final-layer-error-0-73/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-one-layer-injection-budgets-layers-1015-attribute-8086-error) · [Copy provenance](../../../PROVENANCE.json)

