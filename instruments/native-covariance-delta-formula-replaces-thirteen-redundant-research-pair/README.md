# Native covariance-delta formula replaces thirteen redundant research pair-map calls per network

2026-09-25T03:15Z · tool · informational · measured

The research covariance-knob hook repeated the entire pair map, while the native estimator used a linear delta. Porting that formula removes the redundant work with measured rounding-level agreement.

Recorded measurement:

    - 2026-09-24 20:15 [engineer] AUDIT (E-65): native S18 is lean (liveness probe on the torch card, net 37: Cpre, star s1, young push, reads fully read; only old-frame push dead rows ~-0.1% and _bd_update dead/unselected entries ~-0.05% recoverable; L15 already row-cut natively): no native build. TORCH FIND: gn3 'cov_pm' knob hook reruns the whole pair map at L2-14 (13 of 27 calls/net). fable_engineer_v1/gn_linear_hook.py = the native _gn_dcov linear correction in torch: predictions 1e-7 relative, MSE +0.03/-0.00% (37/39), gradients ~1e-3 median (float32 order), refit step -16..-20% (alternating, 3 threads). Offered to the lead for the refit owners.

## Programs

- [code/gn_linear_hook.py](code/gn_linear_hook.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L599](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L599)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../graph/index.html#node=2026-09-25-native-covariance-delta-formula-replaces-thirteen-redundant-research-pair) · [Copy provenance](../../PROVENANCE.json)

