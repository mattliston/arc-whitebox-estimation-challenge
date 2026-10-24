# Full-tensor recursion instruments need independent term and convention checks

2026-08-29 · lesson · informational · measured

The full third-cumulant recursion provided a useful deterministic plane-error instrument and rejected several unstable higher-order variants. Its historical transport claim was later corrected, so this record establishes the measured August instrument behavior rather than a proof that every transport term was present.

Recorded measurement:

    Instrument (2026-08-29): `public_g19_attribution_v1/run_tensor_recursion_v2b.py` carries the FULL 1024³ third-cumulant tensor layer-by-layer on MPS (f16 storage, f32 chunked 3-mode congruence GEMMs; f32 storage measured bit-equivalent). Per-layer ReLU map: F3 = E[f'f'f'] full-ρ transport on distinct entries + CONNECTED plane-feed E[f''f'f] + Gaussian completion (degree-organized Kibble, DTOT 6) + validated M21-stack emission on plane slices + exact univariate diag. Seed exact (pre-0 Gaussian, T=0).
    
    Results (rank 35, vs 8M MC truth with floors quoted): beats the deployed closure's carried K21pre at EVERY layer (L4 6.2% vs 8.7%, L12 11.9% vs 12.5%). Delta-quality q = corr(K̂−closure, truth−closure): 0.70 L4 → 0.45 L11-14. Physics family CONVERGED there: v5 marginal-Edgeworth columns DIVERGE (double-count content the tensor's diag already transports — completion must stay Gaussian-columned); v6 g4-only columns: plane RMS −0.2%, q +0.003; v3 naive two-vertex κ4 (Σκ3κ3/v with tensor squares) explodes 16×→NaN.
    
    **Why:** the instrument is the best deterministic ΔK21 estimator built, and its ceiling is measured — q 0.45-0.7 vs the 0.98 the deployed q² arithmetic demands; in-loop consumption is co-adaptation-blocked (−1%); chain consumption −3-4% endpoint.
    
    **How to apply:** don't re-attempt the closed variants. The instrument ports to width 256 (trivial cost) as an independent cross-check of the R0C1 A2f engine. Floors: [[plane-truth-noise-floors]]. Laws: [[hermite-coefficient-and-connectedness-laws]], [[closure-coadaptation-basin-law]] (extension: better-RMS-but-DIFFERENT errors lose the closure's self-cancellation — in-loop supply needs near-truth, not merely better).

## Programs

- [../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)
- [run_tensor_recursion_v2b.py](run_tensor_recursion_v2b.py)
- [../../../submissions/328326/estimator.py](../../../submissions/328326/estimator.py)

## Evidence

- [run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)

[Research graph](../../../graph/index.html#node=2026-08-29-full-tensor-recursion-instruments-need-independent-term-convention-checks) · [Copy provenance](../../../PROVENANCE.json)

