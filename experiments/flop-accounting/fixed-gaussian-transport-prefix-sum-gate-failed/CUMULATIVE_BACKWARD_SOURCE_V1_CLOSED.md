# Fixed Gaussian/transport prefix sum: gate failed

Phase2, September5,2026. Small four-step dense identity below1.4e-17;
rank35 source0..12, same frozen target cell, current host byte-identical.
No new MC, state writes, fitting, source/age selection, or score.

Fixed natural sum gives weighted signal error ratio .926152227, cosine
.279422474, correction/error norm .214390592. Independent ratios
.932838256/.923361476. Unweighted ratio .820999305. The .65/.65/.70
cost-motivated gates FAIL. Source slice norm .5058224.8.47s, peak4.776GB.
Do not build a sidecar, tune coefficients/ages, or fit this specified sum.
Receipt: cumulative_backward_source_r35_l12_v1.json/.npz.

This sum deliberately omitted the non-Gaussian heavy two-edge source
and is not a verdict on the complete augmented expansion. The old
complete_heavy_path_birth_v1.py also did not implement that contribution
linearly: it introduced the explicit squared-shift remainder. Their
specific failures cannot be called an exact first-order source result.
