# Final signed-pair cohort information — one matched pilot

Phase2, September5,2026. No state mutation. The old G22 mechanistic31
experiment had birth-age channels but ZEROED the final layer. The current
Q sees total scalar third cumulant, not the birth-resolved signed-pair
contributions. This tests that specific missing final information; it is
not a repeat of the old host's age features or scalar birth-gain fitting.

Each cohort has equal/opposite cube weights and propagated factors p+,p-.
With u=(p++p-)/2 and d=p+-p-, exactly
lambda*(p+^3-p-^3)=3*lambda*u^2*d+lambda*d^3/4, entrywise.
At the final scalar read only, expose the sums of BOTH terms for EVERY
birth layer0..14, plus the existing coordinate-star and radial-trace
scalar reads:32 channels normalized by the current sigma^3. These
separate transported mixed and pure terms; no age selection, altered
factor, fitted carrier coefficient or discarded column. All32 sum to
the existing final third cumulant. Birth boundaries are recorded as
shape-only metadata during the unchanged prefix.

Potential cost is only final elementwise signed-pair reads and a158-input
head instead of126: no extra dense matrix product or full covariance
map. A conservative scalar read bound is O(20*n*R) double-precision
primitive operations at the final boundary, <1% of the ~490B bank for
the existing R~25k–30k. This is a bound, not a deployment-meter receipt.

Prerequisites: synthetic exact signed-pair identity <=1e-12 relative;
rank35/100 and every subsequent row must have byte-identical original
full prediction/h/host/sigma/head output/variance, correct paired signs
and complete15 birth boundaries; scalar-source sum relative<=1e-10.
No label access until all target-blind row prerequisites pass.

Then exactly24 current on-policy rows35..58,20 train and heldout37/39/42/48.
Same-size zero-extra control against32-source arm, both initialized from
the frozen current Q with zero weights on new input columns; P frozen.
Seed930065,40 epochs, AdamW5e-4 cosine5e-5, weight decay2e-5, batch8192,
clipnorm5. Train-only center/std (floor1e-6), clip standardized inputs8.
Final epoch only; original1e9 means and actual clipped NumPy output.
Require pooled raw<=.90 of bank and<=.92 of zero-extra control,3/4 bank
wins, independent NumPy/MPS head discrepancy<=2e-9. Failure closes this
32-channel final-only pilot; no200fit, age/component subset, epochs,
learning-rate/seed/range sweep or production implementation.

At most two nice10 two-thread CPU row workers, then one MPS fit. AP Torch
never imports threadpoolctl. Preserve all records, no forum posts,
commits, frozen edits or unqualified submissions.

