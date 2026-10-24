# Shared scalar layer-amplitude calibration: sensitivity gate closed

September5,2026, Phase2. Small action certificate passed: repeated-plane
errors below1.4e-15 and distinct-only action errors below1.8e-15. Both
current-bank baseline trajectories were byte-identical. Fixed central
differences+-.02 covered14 layer coefficients on ranks35/100, two nice10
workers with3BLAS threads,~189seconds each. No training labels entered the
response generation.

One shared bounded(+-.10) least-squares coefficient vector, using the two
existing1e9 development targets, predicts raw ratios .972988/.985448,
only2.70%/1.46% improvement. The optimum is interior (offsets roughly
-.048..+.010), not artificially limited by the bound. This fails the
predeclared20%-on-each information gate. No nonlinear candidate check,
24-net fit, amplitude/window sweep, panel or upload was run. These are
linear-response predictions on development nets, NOT validated gains.

Receipts: distinct_birth_amplitude_certificate_v1.json,
distinct_birth_sensitivity_r35_v1.{json,npz}, rank100 counterpart,
distinct_birth_sensitivity_verdict_v1.json. Preserve them all.
