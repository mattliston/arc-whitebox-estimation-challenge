# Fourth-order plane shortfall aligned with the pairing plane but scalar changes destabilized recursion

2026-09-04 · finding · informational · measured

THE CARD'S FOURTH-ORDER PLANES ARE A ONE-PARAMETER OBJECT (r35, RETRO_PAIRING_ALPHA override in `_reset_stars`; k4dump_alpha*): alpha := 1.0 -> explosion (fourth-cumulant two-pair plane (K22) ratio 830-4047; alpha's natural scale is ~1e-3); alpha x0.5 -> K22 / fourth-cumulant triple-single plane (K31) / k4 ratios 0.15 / 0.12 / 0.14 at L8 (from 0.58 / 0.44 / 0.58), host 2.5e-7; alpha := 0 -> 0.05 / 0.07 / 0.05, host 3.0e-7. So the card's fourth-order planes are essentially alpha x (v v^T + 2 C∘C) (corr 0.999 with the pairing plane): a per-layer least-squares scalar fitted on the pair series' own k22 target, which converges self-consistently to 40-60% of the truth (the k4 table = the alpha correction; the "closure-coadaptation basin"). The engine's plane chain (no such projection) reaches K22 0.92-0.98 -> the deployable fix is the plane-chain port (in-span for the board).

Recorded measurement:

    - T-85 THE CARD'S FOURTH-ORDER SHORTFALL IS ALONG THE PAIRING PLANE (r35): corr(truth K22 - card K22, v v^T + 2 C∘C) = +0.89 / +0.91 / +0.90 at L6 / L10 / L14 (79-82% of the missing energy along it); the card's K22 is 0.999-correlated with that plane, the truth's 0.96-0.985; the k4 shortfall is +0.91 correlated with v^2. => the card's `_reset_stars` least-squares pairing_alpha projection removes the connected content's component along the disconnected plane (the same single defect behind the k4 table 0.99 -> 0.40 and the K22 0.92 -> 0.41). Testing an alpha override (1.0 / half / 0). Partial-scale kurtosis with the operator ON: truth k4 at mix 0.5 r35 3.37e-8 (+25%), r100 4.38e-8 (+57%) - the fourth-order channel is fully owned by the operator, so an in-card fix pays on the board only through better third-order alignment (k3/K21 -> the variance consumer), not directly. (The k4-table "mix" arm was invalid: the rescale path ignores RETRO_FORCE_MIX.)
    
    - T-85b THE CARD'S FOURTH-ORDER PLANES ARE A ONE-PARAMETER OBJECT (r35, RETRO_PAIRING_ALPHA override in `_reset_stars`; k4dump_alpha*): alpha := 1.0 -> explosion (K22 ratio 830-4047; alpha's natural scale is ~1e-3); alpha x0.5 -> K22 / K31 / k4 ratios 0.15 / 0.12 / 0.14 at L8 (from 0.58 / 0.44 / 0.58), host 2.5e-7; alpha := 0 -> 0.05 / 0.07 / 0.05, host 3.0e-7. So the card's fourth-order planes are essentially alpha x (v v^T + 2 C∘C) (corr 0.999 with the pairing plane): a per-layer least-squares scalar fitted on the pair series' own k22 target, which converges self-consistently to 40-60% of the truth (the k4 table = the alpha correction; the "closure-coadaptation basin"). The engine's plane chain (no such projection) reaches K22 0.92-0.98 -> the deployable fix is the plane-chain port (in-span for the board). K21 ERROR STRUCTURE (truth - card, off-diagonal, corr with candidate planes at L6 / L10 / L14): v m^T +0.34 / +0.45 / +0.48, v k3^T +0.35 / +0.45 / +0.47 (row-variance weighting), the card's own K21 +0.23-0.30, C / C∘C / K22 / K31 / m v^T ~ 0: no single simple plane explains it - the skew-plane error is distributed (consistent with its rank structure). HONEST STATE for the 2e-8 bar: the consumer route needs K21 at ~3% (card 8-13%, best derived ~9%); the (dv, dk3) route needs alignment >~ 0.9 on every net (engine 0.6-0.8; tape 0.45). No candidate.

## Evidence

- [LANE_g3_tracker.md#L727](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L727)
- [LANE_g3_tracker.md#L728](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L728)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-fourth-order-plane-shortfall-aligned-pairing-plane-but-scalar) · [Copy provenance](../../../PROVENANCE.json)

