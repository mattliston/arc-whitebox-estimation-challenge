# Corrected annealed kernel law; original numerical failures retained

September6,2026, Phase2. Pure analytic audit; no network sample, fit,
candidate, panel or submission. The old quadrature package is unedited.

The old theory_branching.py applies the centered normalized one-layer
ReLU correlation map repeatedly. The actual network has no activation-
centering operation between layers. Its ideal NNGP raw correlation is the
iterated UNcentered arc-cosine map; subtract the input-wise constant kernel
component only after this composition. See the separate design for the
formula and its infinite-width/high-input-dimensional qualification.

Two Cauchy FFT grids/radii independently recover the low64 coefficients
to1.471e-12 absolute agreement. Degree1 agrees with an independent chain-
derivative calculation to floating-point precision; coefficients are real
and nonnegative at the tested resolution.

| Depth | Old claimed linear fraction | Corrected ideal fraction | Archived rank35 measured fraction |
|---|---:|---:|---:|
|1|.733471|.733471|.733735|
|2|.537980|.595647|.595770|
|6|.155703|.358241|.358668|
|16|.007017|.188797|.222295|

At depth16, corrected cumulative mass through degrees2/4/8/16/32/64 is
.341632/.524462/.704019/.842043/.926548/.969303. Mean degree is14.1756,
not the old approximately460. The correct raw-map branching process has
offspring0 and mean1; conditioning on nonzero final degree happens at the
END. The old no-zero-offspring window argument does not transfer.

Thus most of the old huge annealed/real-net discrepancy is a centering
mistake, not evidence of finite-width coherence. A remaining discrepancy
exists, especially at late layers; this ideal kernel does not quantify
finite-width or finite-input Gaussian-radius corrections. The n64/d6
archived ensemble mean.39656 also exceeds corrected ideal.35824; do not
erase that finite-size/measurement comparison either.

The archived empirical sampling risks, low-degree capture measurements,
and tested MC/affine-CV/antithetic/lattice/subspace failures are unchanged.
This correction does NOT demonstrate a competitive quadrature method or
authorize rerunning those families. The old broad "no tractable surrogate
exists" language was not proved by these experiments; the bounded negative
results are still real. Receipt:nngp_centering_audit_v1.json.

Primary raw-kernel reference:
https://arxiv.org/abs/1711.00165 (Lee et al., Deep Neural Networks as Gaussian
Processes); implementation corroboration:
https://github.com/google/neural-tangents/blob/main/neural_tangents/_src/stax/elementwise.py .
