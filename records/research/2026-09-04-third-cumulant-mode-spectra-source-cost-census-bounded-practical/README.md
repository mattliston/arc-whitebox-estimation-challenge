# Third-cumulant mode spectra and source-cost census bounded practical compression

2026-09-04 · finding · rejected · measured

(census + rescoring, zero new compute): deployed depth-4 third-cumulant arena estimator (D11) card census at util 0.600: third-cumulant canonical-polyadic carrier (CP3) arena 44.9% (near the CP physics floor: ~120k factor pushes x2 = 23% of budget), fourth-order machinery 20.6% (quartic push 13.7 + fourth-cumulant canonical-polyadic carrier (CP4) reads 13.0 + CP4 push 4.7 + g_four 3.0, of util), covariance push (_predict) 10.1%, cubic star 9.1%. Lever (b) "fix the fourth-order planes" is DEAD by the old g29 runs (1e9 frame, fitted transport-energy residual correction (F2J)-off host): CP4 cap 2048 -> 4096 -> 8192 gives 7.85e-8 -> 8.23e-8 -> 8.39e-8 at util 0.627 -> 0.666 -> 0.762 (no gain, more cost); the fourth-cumulant two-pair plane (K22) engine chain gives 4.46e-8/5.04e-8 at 0.618 but is in-span of the closure (T-118 late). CP3 cap ladder with the chain: 8192 4.6e-8, 4096 3.1e-7, 1024 1.56e-6 (util 0.39): the third-order rank is load-bearing.

Recorded measurement:

    - T-126 (tucker_spectrum_test.py, zero billed, r35, 2x16k samples, unbiased cross-Gram): the TRUE post third-cumulant tensor's Tucker mode spectrum: top-64 modes hold 58/79/87% of the energy (L3/L8/L13), top-256 90/96/98%, top-512 98/99.4/99.8% (negative-eigen noise mass <1%). Plane spectrum top-64: 70/91/96%. Moderate rank: 256-512 modes needed for the ~3-5% rms the mean needs -> no trivially cheap exact carry; but old cohorts collapse (Lyapunov), consistent with the spectrum sharpening with depth. In the CP frame the physics minimum is ~1 GEMM push + ~1 GEMM plane read per cohort per layer (13 cohorts at L15 -> ~0.25-0.3 util); our arena+reads bill ~0.35. The exact third-order carry is near its cost floor in this representation; compression buys ~25%.
    
    - T-126b (rescoring the g29 truth-supply runs in the 1e9 frame): host with TRUTH fourth-order planes 2.49e-8 / 2.68e-8 (r35/r100); + truth per-unit k3 1.74e-8 / 1.92e-8; + truth k4 2.13e-8 (r35). The competitor's 1.85e-8 = our card's physics with CORRECT fourth-order supplies, at 1/3 the cost. Reading: the competitor has the same third-order physics done lean (~0.2 util) plus a correct fourth-order plane machinery (ours: CP4 cap defect, k4 13-63% low). Path = same physics, better execution: (a) lean util of the exact carry (0.6 -> ~0.35-0.4), (b) correct fourth-order planes (raw 2.9 -> ~2.1-2.5e-8). Product ~0.4 x 2.3e-8 = 9e-9.
    
    - T-126c (census + rescoring, zero new compute): D11 card census at util 0.600: CP3 arena 44.9% (near the CP physics floor: ~120k factor pushes x2 = 23% of budget), fourth-order machinery 20.6% (quartic push 13.7 + CP4 reads 13.0 + CP4 push 4.7 + g_four 3.0, of util), covariance push (_predict) 10.1%, cubic star 9.1%. Lever (b) "fix the fourth-order planes" is DEAD by the old g29 runs (1e9 frame, F2J-off host): CP4 cap 2048 -> 4096 -> 8192 gives 7.85e-8 -> 8.23e-8 -> 8.39e-8 at util 0.627 -> 0.666 -> 0.762 (no gain, more cost); the K22 engine chain gives 4.46e-8/5.04e-8 at 0.618 but is in-span of the closure (T-118 late). CP3 cap ladder with the chain: 8192 4.6e-8, 4096 3.1e-7, 1024 1.56e-6 (util 0.39): the third-order rank is load-bearing. The competitor's 1.85e-8 equals our card's ORACLE-supply level (truth planes + truth k3: 1.74e-8/1.92e-8): they have something structurally better than a closure-corrected moment card, not just leaner code. Util lever left: compress OLD cohorts (Lyapunov collapse; ~55% of the arena's pushes) by randomized projection ~ -11% util -> ~1.44e-8 adjusted alone. My 35-40% estimate was too high.

## Programs

- [tucker_spectrum_test.py](tucker_spectrum_test.py)
- [../../../experiments/flop-accounting/differentiable-gaussian-closure-four-compensator-designs-stopped-6668-6/code/diff_card.py](../../../experiments/flop-accounting/differentiable-gaussian-closure-four-compensator-designs-stopped-6668-6/code/diff_card.py)

## Evidence

- [LANE_g3_tracker.md#L927](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L927)
- [LANE_g3_tracker.md#L928](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L928)
- [LANE_g3_tracker.md#L929](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L929)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-third-cumulant-mode-spectra-source-cost-census-bounded-practical) · [Copy provenance](../../../PROVENANCE.json)

