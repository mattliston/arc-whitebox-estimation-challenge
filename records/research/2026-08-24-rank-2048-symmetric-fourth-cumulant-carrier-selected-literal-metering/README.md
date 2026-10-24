# Rank-2048 symmetric fourth-cumulant carrier selected for literal metering with pair-series order 32

2026-08-24T04:34Z · decision · adopted · derived

A design audit selected a signed symmetric rank-2048 canonical-polyadic fourth-cumulant carrier (CP4), whose diagonal, triple-single and two-pair planes were read after weight transport. Its discarded planes were returned to the coherent fourth-order Hermite residual, while the bivariate pair-series order remained 32. The frozen complete closure later metered at 2.180585T FLOPs.

Recorded measurement:

    projected arithmetic margins under the stored-marginal reread law: cap 2560 17.669B, cap 2688 8.259B; available headroom 289.821B; independent cost audit of the cap-2048 carrier 234.508B FLOPs; complete frozen V1 2.180585T FLOPs; literal safety margin 18.439B FLOPs; h1 initialisation lambda_i = (3/4 - 1/pi - 3/(2 pi^2)) variance(z1_i)^2

## Programs

- [../../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/estimator.py](../../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/estimator.py)
- [../../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/resource_ledger.py](../../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/resource_ledger.py)

## Evidence

- [FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md#L10](../../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md#L10)
- [FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md#L23](../../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md#L23)
- [FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md#L43](../../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md#L43)
- [FULL_BUDGET_HIGH_ORDER_ARCHITECTURE_STATUS.md#L17](../2026-08-24-full-budget-closure-design-six-mechanisms-projected-1-9092t/FULL_BUDGET_HIGH_ORDER_ARCHITECTURE_STATUS.md#L17)

<details>
<summary>All result files and records (2)</summary>

- [FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md](../../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md) · record
- [FULL_BUDGET_HIGH_ORDER_ARCHITECTURE_STATUS.md](../2026-08-24-full-budget-closure-design-six-mechanisms-projected-1-9092t/FULL_BUDGET_HIGH_ORDER_ARCHITECTURE_STATUS.md) · record

</details>

## Related work

- enables: [Full-budget closure design: six mechanisms projected at 1.9092T FLOPs; dense third-order transport excluded](../2026-08-24-full-budget-closure-design-six-mechanisms-projected-1-9092t/README.md)

[Research graph](../../../graph/index.html#node=2026-08-24-rank-2048-symmetric-fourth-cumulant-carrier-selected-literal-metering) · [Copy provenance](../../../PROVENANCE.json)

