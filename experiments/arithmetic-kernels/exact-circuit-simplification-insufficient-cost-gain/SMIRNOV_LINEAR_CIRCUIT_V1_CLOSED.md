# Exact circuit simplification: insufficient cost gain

September 5, 2026, 20:35 PDT. Phase 2 only. No estimator or upload.

The signed/scaled-wire compiler passes all162 exact rational basis
products. Its materialized scalar linear counts are52/111/153, versus
the primary shared program's62/123/149. No signs or scales are assumed
free when materialized: input-product gauge factors are carried into
the output circuit, and final output negations/scalings are charged.

Adding the new circuit and transposition-principle arithmetic counts to
the existing exact-fringe mixed-FMM DP gives optimistic total savings:

- rank35:20.07595% unrestricted;17.81495% with necessary leaf A/B/C
  storage bounded by the existing1,168,900,096-byte arena.
- rank100:20.28315% unrestricted;17.90701% with that memory lower bound.

Copies, allocation, complete live memory, numerical stability and remote
runtime are still unpriced. These are not achieved FLOPs or score gains.
The preregistered25%-on-both production-design gate fails. Do not build
this kernel merely because the unconstrained estimate crosses4e-9.

Certificates and receipts: smirnov_linear_circuit_v1.json and
smirnov_linear_circuit_envelope_v1.json. The primary data is preserved
unchanged; no downloaded source was executed.
