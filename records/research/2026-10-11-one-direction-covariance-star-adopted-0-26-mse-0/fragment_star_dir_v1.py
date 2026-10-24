
# BEGIN OPUS-ENGINEER a62 ONE-DIRECTION STAR (fragment_star_dir_v1, 2026-10-11; a physics change: the star's transport keeps its
# row-sum part exactly, as the fired form does, plus the star plane's leading rank-one piece along the row sums: SS v3's own form -1,
# s2 = ((W o W)' A' a) (W' a)' / a'a with a = A 1, 8 n^2 flops a layer). Appended after the release fragments; it only rebinds the
# per-layer switch that _ss_factors reads at call time (one entry for the star moved into each of the layers 1..15).
# Native, 16 fresh networks 7100136-151 against the fired S47e: final-layer MSE -0.26% (se 0.11, 11 of 16 lower), bill +0.035%.
_SS_RANK = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
# END OPUS-ENGINEER a62 ONE-DIRECTION STAR
