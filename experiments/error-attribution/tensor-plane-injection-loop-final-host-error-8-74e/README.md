# Tensor-plane injection: in-loop final host error 8.74e-8 versus baseline 8.83e-8

2026-08-29 · experiment · informational · measured

Tensor-recursion consumption measured on rank35: (1) delta quality q = corr(K_tensor - closure, truth - closure): 0.70 peak fourth-cumulant Edgeworth operator (L4), 0.40-0.47 at L11-14 (best deterministic dK21 estimator built, no fitting). (2) IN-LOOP forcing of tensor planes: L16 8.74e-8 vs base 8.83e-8 (-1 pct only, vs -33 pct for truth) and downstream planes WORSEN - the co-adaptation law: the closure cancels its OWN errors downstream; different smaller errors lose the cancellation. In-loop supply requires near-truth. (3) Outside-loop chain: dv_corr 0.42/0.39/0.26 at L12/14/15 (vs 0.78-0.84 truth-fed) -> approx -3-4 pct endpoint. (4) v3 two-vertex kappa4 self-supply EXPLODED (16x L2 -> NaN): f16 tensor-squared noise accumulation + propagator subtleties; needs f32 + the receipt's local form. VERDICT: carrier line delivers real but small value; q needed 0.98, ceiling of current physics ~0.7-0.8.

Recorded measurement:

    Tensor-recursion consumption measured on rank35: (1) delta quality q = corr(K_tensor - closure, truth - closure): 0.70 peak L4, 0.40-0.47 at L11-14 (best deterministic dK21 estimator built, no fitting). (2) IN-LOOP forcing of tensor planes: L16 8.74e-8 vs base 8.83e-8 (-1 pct only, vs -33 pct for truth) and downstream planes WORSEN - the co-adaptation law: the closure cancels its OWN errors downstream; different smaller errors lose the cancellation. In-loop supply requires near-truth. (3) Outside-loop chain: dv_corr 0.42/0.39/0.26 at L12/14/15 (vs 0.78-0.84 truth-fed) -> approx -3-4 pct endpoint. (4) v3 two-vertex kappa4 self-supply EXPLODED (16x L2 -> NaN): f16 tensor-squared noise accumulation + propagator subtleties; needs f32 + the receipt's local form. VERDICT: carrier line delivers real but small value; q needed 0.98, ceiling of current physics ~0.7-0.8. GATE for next block: if kappa4-f32 + second-order feeds do not push q(L12+) past 0.8, the carrier line is capped -> pivot R0C1.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_g19_attribution_v1/run_force_k21post.py](code/public_g19_attribution_v1/run_force_k21post.py)
- [code/public_g19_attribution_v1/run_tensor_q.py](code/public_g19_attribution_v1/run_tensor_q.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [TENSOR_Q_rank0035.json#L1](../../flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/TENSOR_Q_rank0035.json#L1)
- [FORCE_K21_rank0035_tensor.json#L1](FORCE_K21_rank0035_tensor.json#L1)

<details>
<summary>All result files and records (2)</summary>

- [TENSOR_Q_rank0035.json](../../flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/TENSOR_Q_rank0035.json) · result
- [FORCE_K21_rank0035_tensor.json](FORCE_K21_rank0035_tensor.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-29-tensor-plane-injection-loop-final-host-error-8-74e) · [Copy provenance](../../../PROVENANCE.json)

