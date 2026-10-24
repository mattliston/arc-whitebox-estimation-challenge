# Hermite coefficients use standardized axes and cumulant responses require connectedness

2026-08-29 · lesson · informational · measured

Dividing standardized Hermite coefficients by an additional power of the standard deviation double-counted their scaling. Connected cumulant responses also require cancellation of disconnected spectator terms; omitting that cancellation caused depth divergence.

Recorded measurement:

    Two laws from the 2026-08-29 tensor-recursion build (`public_g19_attribution_v1/run_tensor_recursion_v2b.py`), each validated by an independent instrument after silently corrupting a full construction:
    
    1. **Coefficient law.** In Kibble/Mehler expansions over standardized Gaussians, the axis coefficients are `a[t] = E[f(m+σz)·He_t(z)]` — the c1-columns AS THEY ARE (`c1[q] = σ·He_{q−2}(−α)·pdf`, q≥2). Dividing by σ^t double-scales and produced 20-80% error vs exact GH quadrature; with the fix, 0.2-2.5%. Stein shifts: `E[f' He_t] = He_{t−1}(−α)·pdf`, `E[f'' He_t] = He_t(−α)·pdf/σ`. Validate any new column set against `edg2_relu_mean` at t=0 (matched to 0.0) and 3D quadrature before use.
    
    2. **Connectedness cancellation.** The response of a CONNECTED cumulant to an upstream cumulant insertion subtracts the responses of its disconnected products. Concretely: κ3pre[i,i,j] feeding distinct cum3post[i,j,k] via E[f''f'f] has its t=0 (no-bond-to-spectator) term EXACTLY cancelled — a spectator with no ρ-bond cannot join a joint cumulant. Implementing the raw (unfiltered) feed diverged the recursion 19× by depth; the connected filter (spectator degree ≥ 1) restored it and beat the closure.
    
    **Why:** both errors pass unit-level eyeballing and only show up at aggregate scale; each cost hours.
    
    **How to apply:** any Edgeworth/Mehler/Kibble machinery: (a) never σ^t-divide the t-index; (b) every response/feed term must carry the connected filter; (c) gate new diagram code on GH quadrature per-element AND a known-map tie before trusting aggregates. Related: [[tensor-recursion-instrument]], [[closure-coadaptation-basin-law]].

## Programs

- [../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)
- [../2026-08-29-full-tensor-recursion-instruments-need-independent-term-convention-checks/run_tensor_recursion_v2b.py](../2026-08-29-full-tensor-recursion-instruments-need-independent-term-convention-checks/run_tensor_recursion_v2b.py)
- [../../../submissions/328326/estimator.py](../../../submissions/328326/estimator.py)

## Evidence

- [run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)

[Research graph](../../../graph/index.html#node=2026-08-29-hermite-coefficients-use-standardized-axes-cumulant-responses-require) · [Copy provenance](../../../PROVENANCE.json)

