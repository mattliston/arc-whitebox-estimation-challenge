# Exact mean-formula inputs remove 95% of final-layer error across all layers and 20% at layer 15

2026-10-01T19:20Z · finding · informational · measured

On the torch card over four networks, replacing the per-unit variance, skew and kurtosis inputs of the mean formula by their Monte Carlo truth at every layer with the learned mean correction off takes the final-layer MSE from 17.2 to 0.8 (x1e-9); with the learned correction on and no refit the three together give -67%, variance alone -34%, kurtosis -7%, skew 0%. Made exact at layer 15 only the gain is -20% (variance -8.5%, kurtosis -4.4%, skew -2.2%). The learned correction is worth 13% today and about 4 more when the inputs are exact.

Recorded measurement:

    final-layer MSE x1e-9: learned off, all exact 17.2 -> 0.8 (-95%); learned on, no refit: all three 4.9 (-67%), variance 9.9 (-34%), kurtosis 13.9 (-7%), skew 15.0 (0%); layer 15 only: 15.0 -> 12.0 (-20%), variance -8.5%, kurtosis -4.4%, skew -2.2%, variance+kurtosis -15%; J2W error layers 0-14 = 0.99 of ours, final layer 0.87

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L14](../../TEAM_LOG_2026-10-01_to_10-22.md#L14)
- [TEAM_LOG_2026-10-01_to_10-22.md#L20](../../TEAM_LOG_2026-10-01_to_10-22.md#L20)

<details>
<summary>All result files and records (3)</summary>

- [a10_final_layer_oracle.md](a10_final_layer_oracle.md) · result
- [RESULT.md](../2026-10-01-torch-research-estimator-lowers-final-layer-error-0-73/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-exact-mean-formula-inputs-remove-95-final-layer-error) · [Copy provenance](../../../PROVENANCE.json)

