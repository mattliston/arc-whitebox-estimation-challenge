# Late-layer state errors and low-rank covariance residuals need separate value tests

2026-09-03 · lesson · informational · measured

Oracle state repairs localized much of the host error to the last layers, and covariance residuals showed a strong low-rank mean-aligned component. Directly inserting the true low-rank covariance residual nevertheless failed to reproduce the value of the exact per-unit variance correction.

Recorded measurement:

    Phase-2 (1024x16) laws measured 2026-09-03 (LANE_g3_tracker T-86..T-100d):
    
    - **Score frame.** adjusted = raw x max(util, 0.1). Our bank: 2.94e-8 x 0.6065 = 1.78e-8. Our card's cost floor is ~0.5 (CP3 arena push 0.27 is load-bearing: any smaller cap explodes raw 10x with the sealed arena AND with a plain push; f16 is billed as f32).
    - **Error is born late.** Exact per-unit (v,k3,k4) at layers 13-15 alone removes 49% of the F2J-off host error, layers 9-15 86%, layers 1-4 nothing. Operator kept: half (v,k3) at 9-15 gives -52%; half v alone at 9-15 gives -20%/-29% (r35/r100). Any expensive per-unit supply only needs the last 3-7 layers.
    
    - **Closed with receipts.** Refit on a fixed-plane host (10% removal); transported state corrector (dv held-out 0.19 even with the oracle previous-layer error); low-rank cohort bank (spectra decay, but billed truncations + dense-map reads cost more than they save: 0.84 util); late-window carry; chain card as a bank (~2.05e-8 adjusted).
    - **In-card instruments** (estimator_retro.py): RETRO_K22_CHAIN (the engine's fourth-order plane chain in-card, certified), RETRO_PLAIN_CP3, RETRO_LR_BANK (certified at full rank), RETRO_FORCE_LAYERS (window), RETRO_FORCE=Ck (rank-k covariance write).

## Programs

- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py)

## Evidence

- [estimator_retro.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py)

[Research graph](../../../graph/index.html#node=2026-09-03-late-layer-state-errors-low-rank-covariance-residuals-need) · [Copy provenance](../../../PROVENANCE.json)

