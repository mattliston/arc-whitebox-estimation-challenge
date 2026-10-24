# Coherent state rescaling: fidelity qualified for the bounded response test

2026-09-05,14:14PDT, machine clock. Phase2 only; no accuracy gain yet.

Synthetic complete T/Q scaling algebra passes at widths8/16; exact-arithmetic
errors<3e-16, with f32 stored-factor error<5.83e-8. Zero-action NumPy bank
parity is byte-exact on35/100, including both frozen learned consumers.
CPU zero-action maxima are1.70847e-7/2.23704e-7. Actual secants at amplitudes
.02/.01 agree with independent NumPy within.399%/.358% respectively.

Rank35 passes the complete original dynamic fidelity test: half-step
discrepancy.827%, tangent discrepancies.306%/.773%.

Rank100's original flag remains FALSE. Both positive test steps change two
active memberships atL12; halving the step changes the secant by14.595%,
and the zero-point tangent differs from these actual secants14.892%/29.177%.
The CPU model reproduces the actual selection-changing paths accurately;
it is not justified to regard that branch crossing as a smooth derivative.

The separate fixed-baseline membership AND ordering certificate is byte
exact at zero, has half-step discrepancy.687%, and tangent errors.304%/
.635%, passing the existing5% local gates. This qualifies a local response
solve, not globally smooth dynamics or any deployment gain. All actual
forwards retain dynamic selection. The failed original flag is preserved.

Next bounded test is fully declared in COHERENT_SCALE_INFORMATION_V1.md:
complete12-dimensional response on four fit-side networks; fixed ridge
and radius; require20% predicted fit-side reduction before touching four
holdouts, then20% mean and10% every holdout. Independent nonlinear checks
would still be required. No automatic larger fit or submission.
