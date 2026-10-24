# The fifth-cumulant two-vertex formula is valid only in its verified chaos expansion

2026-08-26 · lesson · informational · measured

The fifth-cumulant relation was derived and verified at leading order in a graded chaos expansion. Its fourth-cumulant feed-down explained only part of the measured deficit, and the same argument did not determine the sixth cumulant for free.

Recorded measurement:

    2026-08-26 (trivariate-deriver, package `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_trivariate_derivation_v1/`, KAPPA5_CLOSURE.md + kappa5_verify.py + KAPPA5_CLOSURE.json).
    
    **The law (one order above [[kappa4-is-a-function-of-kappa3]]):** standardized
    `kappa5 = 5·k3·k4 − (40/9)·k3³` (raw: /v and /v²). Coefficients DERIVED (chaos-graded frame: He2~eps, He3~eps², the same frame under the deployed k4 = Σ K21²/v law), verified exact at leading order (residual slope ~2, −0.02% at eps=0.04). Equivalent: 5·k3·k4_intrinsic + (20/9)k3³ with k4_intrinsic = k4 − (4/3)k3². Bivariate planes: K41 = ρ(3k3ik4i − (28/9)k3i³) + ρ²k3jk4i + ρ³(k3ik4j − (4/3)k3ik3j²); K32 in the doc. kappa6 two-vertex form NOT free (He4 content enters at its own leading order).
    
    **Emission feed-down:** pre-k5 → post-quartic via the He5 column (kappa5/120·c[p,5], Stein: c[4,5] = 24σ⁴φ(w)); dk4_post/dk5 bracket in the doc. He5-alone overshoots at large tilt (He7 k3k4-column opposes — same near-cancellation geometry as the L5 pair finding).
    
    **Why it matters:** at the measured ladder tilts (|g3| 0.1–0.25, g4 0.03–0.07) the closure-fed kappa5 column injects median 0.49%/layer (max 1.27%) vs the ~1.22%/layer needed for the masked-median k4 ratio decay 0.95@L5→0.84@L15 — ratio 0.40: owns roughly a third-to-half of the deficit, NOT all. Sign flips at k4 = (8/9)k3² (cheap layer-by-layer discriminator).
    
    **How to apply:** supply slot = the existing `public_k5_probe_v1` He5 wiring (it lacked exactly the non-Gaussian pre-k5). Do NOT direct-retrofit onto the effective-cumulant host ([[closure-coadaptation-basin-law]], the L5 lambda-scan linear-loss verdict); consume via oracle-frame accounting, joint retune, or clean-slate.

## Programs

- [../../../experiments/flop-accounting/fourth-cumulant-2-1-1-contraction-supply-small-raw/code/public_trivariate_derivation_v1/kappa5_verify.py](../../../experiments/flop-accounting/fourth-cumulant-2-1-1-contraction-supply-small-raw/code/public_trivariate_derivation_v1/kappa5_verify.py)
- [../../../experiments/flop-accounting/fourth-cumulant-2-1-1-contraction-supply-small-raw/code/public_trivariate_derivation_v1/l5_verify.py](../../../experiments/flop-accounting/fourth-cumulant-2-1-1-contraction-supply-small-raw/code/public_trivariate_derivation_v1/l5_verify.py)
- [../../../experiments/flop-accounting/fourth-cumulant-2-1-1-contraction-supply-small-raw/code/public_trivariate_derivation_v1/verify.py](../../../experiments/flop-accounting/fourth-cumulant-2-1-1-contraction-supply-small-raw/code/public_trivariate_derivation_v1/verify.py)

## Evidence

- [kappa5_verify.py](../../../experiments/flop-accounting/fourth-cumulant-2-1-1-contraction-supply-small-raw/code/public_trivariate_derivation_v1/kappa5_verify.py)

[Research graph](../../../graph/index.html#node=2026-08-26-fifth-cumulant-two-vertex-formula-valid-only-verified-chaos) · [Copy provenance](../../../PROVENANCE.json)

