# Tensor-carrier supply correlations 0.47 and 0.40 missed the 0.8 continuation threshold

2026-08-29 · experiment · informational · measured

GATE RESULT: physics ladder on the tensor carrier CONVERGED at q(L12/L14) = 0.47/0.40 - gate was 0.8, goal bar 0.98. v5 (marginal-Edgeworth columns) DIVERGES (0.40 L12 -> 0.82 L15): double-counts diag-tensor content the F3 transport already carries - completion must stay Gaussian-columned (a[0]-check vs edg2 passed exactly; the error is joint-level, not the formula). v6 (g4-only columns, content genuinely outside the cum3 carrier): -0.2 pct plane RMS at L12/13, q +0.003 = nil. Residual deep fresh error is higher-order joint structure outside all implemented diagram families. CARRIER LINE CAPPED for the deployed <1e-8 goal.

Recorded measurement:

    GATE RESULT: physics ladder on the tensor carrier CONVERGED at q(L12/L14) = 0.47/0.40 - gate was 0.8, goal bar 0.98. v5 (marginal-Edgeworth columns) DIVERGES (0.40 L12 -> 0.82 L15): double-counts diag-tensor content the F3 transport already carries - completion must stay Gaussian-columned (a[0]-check vs edg2 passed exactly; the error is joint-level, not the formula). v6 (g4-only columns, content genuinely outside the cum3 carrier): -0.2 pct plane RMS at L12/13, q +0.003 = nil. Residual deep fresh error is higher-order joint structure outside all implemented diagram families. CARRIER LINE CAPPED for the deployed <1e-8 goal. PIVOT EXECUTED -> R0C1 (pure-closure kappa3 ceiling, width 256, gate_a0 Owens-T already passed). The tensor recursion instrument carries over there.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_g19_attribution_v1/run_tensor_q.py](code/public_g19_attribution_v1/run_tensor_q.py)
- [code/public_g19_attribution_v1/run_tensor_recursion_v5.py](code/public_g19_attribution_v1/run_tensor_recursion_v5.py)
- [code/public_g19_attribution_v1/run_tensor_recursion_v6.py](code/public_g19_attribution_v1/run_tensor_recursion_v6.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [TENSOR_RECURSION_rank0035_v5.json#L1](TENSOR_RECURSION_rank0035_v5.json#L1)
- [TENSOR_RECURSION_rank0035_v6.json#L1](TENSOR_RECURSION_rank0035_v6.json#L1)
- [TENSOR_Q_rank0035_v6.json#L1](TENSOR_Q_rank0035_v6.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [TENSOR_RECURSION_rank0035_v5.json](TENSOR_RECURSION_rank0035_v5.json) · result
- [TENSOR_RECURSION_rank0035_v6.json](TENSOR_RECURSION_rank0035_v6.json) · result
- [TENSOR_Q_rank0035_v6.json](TENSOR_Q_rank0035_v6.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-29-tensor-carrier-supply-correlations-0-47-0-40-missed) · [Copy provenance](../../../PROVENANCE.json)

