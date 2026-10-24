# Exact circuit simplification: insufficient cost gain

2026-09-06T03:35Z · experiment · rejected · measured

The signed/scaled-wire compiler passes all162 exact rational basis products. Its materialized scalar linear counts are52/111/153, versus the primary shared program's62/123/149. Copies, allocation, complete live memory, numerical stability and remote runtime are still unpriced.

Recorded measurement:

    The signed/scaled-wire compiler passes all162 exact rational basis products. Its materialized scalar linear counts are52/111/153, versus the primary shared program's62/123/149. No signs or scales are assumed free when materialized: input-product gauge factors are carried into the output circuit, and final output negations/scalings are charged. - rank35:20.07595% unrestricted;17.81495% with necessary leaf A/B/C storage bounded by the existing1,168,900,096-byte arena. - rank100:20.28315% unrestricted;17.90701% with that memory lower bound. Copies, allocation, complete live memory, numerical stability and remote runtime are still unpriced. These are not achieved FLOPs or score gains. The preregistered25%-on-both production-design gate fails. Do not build this kernel merely because the unconstrained estimate crosses4e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/audit_smirnov_executable_transpose_v1.py](code/audit_smirnov_executable_transpose_v1.py)
- [code/audit_smirnov_linear_circuit_v1.py](code/audit_smirnov_linear_circuit_v1.py)
- [code/audit_smirnov_slp_v1.py](code/audit_smirnov_slp_v1.py)
- [code/executable_linear_dag_v1.py](code/executable_linear_dag_v1.py)
- [code/price_mixed_fmm_envelope_v1.py](code/price_mixed_fmm_envelope_v1.py)
- [code/price_mixed_fmm_peel_v2.py](code/price_mixed_fmm_peel_v2.py)
- [code/price_rectangular_catalog_v1.py](code/price_rectangular_catalog_v1.py)
- [code/price_smirnov_linear_circuit_v1.py](code/price_smirnov_linear_circuit_v1.py)

## Evidence

- [SMIRNOV_LINEAR_CIRCUIT_V1_CLOSED.md#L1](SMIRNOV_LINEAR_CIRCUIT_V1_CLOSED.md#L1)
- [SMIRNOV_LINEAR_CIRCUIT_V1_CLOSED.md#L16](SMIRNOV_LINEAR_CIRCUIT_V1_CLOSED.md#L16)

<details>
<summary>All result files and records (4)</summary>

- [SMIRNOV_LINEAR_CIRCUIT_V1_CLOSED.md](SMIRNOV_LINEAR_CIRCUIT_V1_CLOSED.md) · record
- [smirnov_linear_circuit_v1.json](smirnov_linear_circuit_v1.json) · result
- [smirnov_linear_circuit_envelope_v1.json](smirnov_linear_circuit_envelope_v1.json) · result
- [SMIRNOV_LINEAR_CIRCUIT_DESIGN_V1.md](SMIRNOV_LINEAR_CIRCUIT_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-exact-circuit-simplification-insufficient-cost-gain) · [Copy provenance](../../../PROVENANCE.json)

