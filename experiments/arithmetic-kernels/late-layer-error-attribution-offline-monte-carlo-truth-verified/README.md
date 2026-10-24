# Late-layer error attribution vs offline Monte Carlo truth: verified

2026-08-29 · experiment · informational · measured

Which mechanism owns the order-24 Hermite pair-series closure (Q24) host's late-layer (l=10-15) mean error: marginal Edgeworth-truncation at the emission, pair/covariance state error echo, or non-Edgeworth width-finite remainder? Offline Monte Carlo (MC) truth (weights-only, seeded, no sealed access) at per-unit SE below the q scale decides. This is the cheapest fatal gate: the winning mechanism defines which construction (higher-order closure with kappa5/kappa6 laws, late-layer rho/covariance repair, width-finite kernel correction, or late-layer FLOP reallocation) gets the one-arm build.

Recorded measurement:

    Which mechanism owns the Q24 host's late-layer (l=10-15) mean error: marginal Edgeworth-truncation at the emission, pair/covariance state error echo, or non-Edgeworth width-finite remainder? Offline MC truth (weights-only, seeded, no sealed access) at per-unit SE below the q scale decides. This is the cheapest fatal gate: the winning mechanism defines which construction (higher-order closure with kappa5/kappa6 laws, late-layer rho/covariance repair, width-finite kernel correction, or late-layer FLOP reallocation) gets the one-arm build. VERDICT (2026-08-29, both rows, preregistered gates met): INGREDIENTS. The drifted pre-layer state carries the late-layer mean error - term B = 114/121 percent of noise-corrected energy on ranks 35/100; the order-2 emission formula at TRUE state errs BELOW the 8M-sample noise floor (term A negative after noise subtraction); the host-convention term C is zero to 8 decimals (which also validates tap alignment). MC noise 12/17 percent of raw. The formula-upgrade branch (kappa5 column) is dead for this purpose; the fight is state propagation accuracy. Ingredient sub-split (m vs v vs k3 vs k4) running from saved states.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [PREREGISTER.md#L1](PREREGISTER.md#L1)
- [ATTRIBUTION_rank0035.json#L1](ATTRIBUTION_rank0035.json#L1)
- [ATTRIBUTION_rank0100.json#L1](ATTRIBUTION_rank0100.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [PREREGISTER.md](PREREGISTER.md) · record
- [ATTRIBUTION_rank0035.json](ATTRIBUTION_rank0035.json) · result
- [ATTRIBUTION_rank0100.json](ATTRIBUTION_rank0100.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-29-late-layer-error-attribution-offline-monte-carlo-truth-verified) · [Copy provenance](../../../PROVENANCE.json)

