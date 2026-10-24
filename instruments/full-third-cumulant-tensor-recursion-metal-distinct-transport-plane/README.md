# Full third-cumulant tensor recursion on Metal: distinct transport and plane-emission instrument

2026-08-29 · tool · informational · measured

NEW INSTRUMENT + FIRST WIN: full cum3 tensor recursion on MPS (f16 storage, 3-mode congruence push, ~45s/layer, 11 min full net). Physics per layer: F3=E[f'f'f'] full-rho transport on distinct entries (Stein b[t]=He_{t-1}pdf), CONNECTED plane-feed E[f''f'f] (t=0 part cancels by disconnectedness - v2.0 diverged 19x until filtered), Gaussian completion (degree-organized Kibble, coefficient law a[t]=E[f He_t] NO sigma^t - the quadrature-validated fix), analytic emission plane slices, exact seed T_pre0=0. RESULT: beats closure K21pre at EVERY layer (fourth-cumulant Edgeworth operator (L4) 6.2 vs 8.7, L12 11.9 vs 12.5, L13 12.7 vs 12.9 pct). f32=f16 identical.

Recorded measurement:

    NEW INSTRUMENT + FIRST WIN: full cum3 tensor recursion on MPS (f16 storage, 3-mode congruence push, ~45s/layer, 11 min full net). Physics per layer: F3=E[f'f'f'] full-rho transport on distinct entries (Stein b[t]=He_{t-1}pdf), CONNECTED plane-feed E[f''f'f] (t=0 part cancels by disconnectedness - v2.0 diverged 19x until filtered), Gaussian completion (degree-organized Kibble, coefficient law a[t]=E[f He_t] NO sigma^t - the quadrature-validated fix), analytic emission plane slices, exact seed T_pre0=0. RESULT: beats closure K21pre at EVERY layer (L4 6.2 vs 8.7, L12 11.9 vs 12.5, L13 12.7 vs 12.9 pct). f32=f16 identical. ALSO: plane-truth MC noise floors measured (L1 11.7 pct, L2 7.8, L12 1.5) - early-layer closure errors were noise-inflated (L2 true 3.5 pct), deep real. Deep slope (+0.9 pct/layer) shared with closure = missing 4th-order supply (K22/K31 emission inputs). Next = v3: two-vertex kappa4 planes from the tensor's own kappa3 (oracle-parity law) feeding the emission.

## Programs

- [code/public_g19_attribution_v1/run_tensor_recursion_v2.py](code/public_g19_attribution_v1/run_tensor_recursion_v2.py)
- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [TENSOR_RECURSION_rank0035_v2.json#L1](TENSOR_RECURSION_rank0035_v2.json#L1)
- [PLANE_NOISE_FLOORS.json#L1](PLANE_NOISE_FLOORS.json#L1)
- [run_tensor_recursion_v2.py#L1](code/public_g19_attribution_v1/run_tensor_recursion_v2.py#L1)

<details>
<summary>All result files and records (2)</summary>

- [TENSOR_RECURSION_rank0035_v2.json](TENSOR_RECURSION_rank0035_v2.json) · result
- [PLANE_NOISE_FLOORS.json](PLANE_NOISE_FLOORS.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-08-29-full-third-cumulant-tensor-recursion-metal-distinct-transport-plane) · [Copy provenance](../../PROVENANCE.json)

