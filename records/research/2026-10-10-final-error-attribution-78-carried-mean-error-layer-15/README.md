# Final-error attribution: 78% carried mean error; a layer-15-only change has about a 22% ceiling

2026-10-10T04:41Z · finding · informational · derived

Per-unit anatomy against paired 32M truths splits each layer's post-mean error into carried mean, variance, skew and kurtosis channels (93-97% of the energy); the carry gain is 0.85-0.91 per layer, so the final error = local_15 + 0.86 local_14 + 0.74 local_13 + ...

Recorded measurement:

    L15 shares carried / variance / skew / kurtosis 78/9/4/4%; replacing the card's L15 mean by truth cuts final MSE 59-74%, the variance alone 7-12%

## Programs

- [t70_layer_anatomy.py](t70_layer_anatomy.py)
- [pu_cert.py](pu_cert.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L496](../../TEAM_LOG_2026-10-01_to_10-22.md#L496)
- [TEAM_LOG_2026-10-01_to_10-22.md#L507](../../TEAM_LOG_2026-10-01_to_10-22.md#L507)
- [TEAM_LOG_2026-10-01_to_10-22.md#L542](../../TEAM_LOG_2026-10-01_to_10-22.md#L542)
- [TEAM_LOG_2026-10-01_to_10-22.md#L551](../../TEAM_LOG_2026-10-01_to_10-22.md#L551)
- [TEAM_LOG_2026-10-01_to_10-22.md#L552](../../TEAM_LOG_2026-10-01_to_10-22.md#L552)

<details>
<summary>All result files and records (2)</summary>

- [t70_layer_anatomy.txt](t70_layer_anatomy.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Truth variance at layers 12-15: -24.6% final MSE; third and fourth cumulants give -4.2% and -4.8%](../2026-10-10-truth-variance-layers-12-15-24-6-final-mse/README.md)
- enables: [Error after transport-gate correction: two thirds originates at layers 12-15; board transfer about 0.8x](../2026-10-12-error-after-transport-gate-correction-two-thirds-originates-layers/README.md)
- enables: [Skew- and kurtosis-corrected transport gate adopted: -0.348% pooled MSE on 24 fresh networks](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-final-error-attribution-78-carried-mean-error-layer-15) · [Copy provenance](../../../PROVENANCE.json)

