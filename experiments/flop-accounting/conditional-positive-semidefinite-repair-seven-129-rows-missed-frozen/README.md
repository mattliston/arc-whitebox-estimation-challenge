# Conditional positive-semidefinite repair: seven of 129 rows missed the frozen savings threshold

2026-08-25 · experiment · rejected · measured

The 129-row target-free census validated the conditional-repair candidate in substance - every row finite and improving (minimum saving 206,167,895,046 FLOPs), maximum output delta 2.0483e-8 against a 1e-7 bar, every W total under 2^41 - but the frozen per-row floor of 2.6e11 failed on seven rows, so this spelling closed under its one-arm law. The independent v2 audit corrected the mechanism of record: savings equal exactly 22,907,543,894 FLOPs per certified skip with zero Cholesky fallbacks observed; low rows simply have more near-floor layers that never reach the gauge arm.

Recorded measurement:

    The 129-row target-free census validated the conditional-repair candidate in substance - every row finite and improving (minimum saving 206,167,895,046 FLOPs), maximum output delta 2.0483e-8 against a 1e-7 bar, every W total under 2^41 - but the frozen per-row floor of 2.6e11 failed on seven rows, so this spelling closed under its one-arm law. The independent v2 audit corrected the mechanism of record: savings equal exactly 22,907,543,894 FLOPs per certified skip with zero Cholesky fallbacks observed; low rows simply have more near-floor layers that never reach the gauge arm.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v1/run_ab_census.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_condrepair_deployment_card_v1/run_ab_census.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_host_compute_value_attribution_v1/estimator_condrepair.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_host_compute_value_attribution_v1/estimator_condrepair.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [AUDIT_RECEIPT.json#L1](AUDIT_RECEIPT.json#L1)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [CENSUS_RESULT.json#L1](CENSUS_RESULT.json#L1)
- [W_CLOSURE.json#L5](W_CLOSURE.json#L5)

<details>
<summary>All result files and records (5)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [AUDIT_RECEIPT.json](AUDIT_RECEIPT.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [CENSUS_RESULT.json](CENSUS_RESULT.json) · result
- [W_CLOSURE.json](W_CLOSURE.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-conditional-positive-semidefinite-repair-seven-129-rows-missed-frozen) · [Copy provenance](../../../PROVENANCE.json)

