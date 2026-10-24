# Projected plane feed: closed, no transferable raw gain

The small dense all-distinct identity passes(max absolute4.34e-19), and
the Frobenius projection normal equation passes(absolute1.78e-15).

The v1 diagnostic initially stopped because it required nonnegative birth
coefficients even on units that emit no new factors. The versioned v2 guard
checks the actual selected birth support. Both full trajectories confirm
zero negative selected coefficients; negative unselected coefficients emit
no factors. No coefficient was clipped, tuned, or changed by that guard fix.

Unlearned current full-pair host, stored1e9 post targets:

- Rank35:2.40240165536e-8 versus2.46412689534e-8, -2.5050%.
- Rank100:2.79269415853e-8 versus2.69882407107e-8, +3.4782%.

The fixed projection fails the5%-on-both development gate. It captures only
about18%-32% of its own weighted one-bond source energy at deep layers,
not a measured fraction of the actual estimator error. No larger source
bank, strength sweep, matched training, port, panel, or submission follows.
Receipts:`projected_plane_feed_certificate_v1.json` and
`projected_plane_feed_r35_v2.json`/`projected_plane_feed_r100_v2.json`.
