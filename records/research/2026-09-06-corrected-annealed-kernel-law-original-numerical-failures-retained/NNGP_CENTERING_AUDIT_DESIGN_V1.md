# Read-only correction of the old annealed kernel law

September6,2026, Phase2. No estimator experiment, sampling or fit.
The old public_quadrature_derivation_v1/theory_branching.py composes the
centered scalar ReLU correlation map at every layer. That would require
centering the activations before each weight matrix, which the actual
bias-free ReLU network does not do. Its "finite-width coherence" explanation
of the theory/measurement gap therefore needs correction.

For normalized He ReLU, raw covariance K follows

    A(t) = [sqrt(1-t*t)+(pi-acos(t))*t]/pi,
    K_L(t) = A composed L times (t).

The centered, normalized across-input kernel in the joint infinite-width/
high-input-dimension approximation is

    B_L(t) = [K_L(t)-K_L(0)]/[1-K_L(0)].

This is NOT an exact claim about a single fixed width1024 net or finite-
dimensional Gaussian radial fluctuations. The input-wise constant part
is removed only after composing the raw kernel. Consequently
B_L'(0)=product_{j=0..L-1}A'(K_j(0))/(1-K_L(0)), and
B_L'(1)=1/(1-K_L(0)), because A'(1)=1.

Compute the low64 Taylor coefficients by two independent Cauchy FFT
discretizations (radius.85/grid2048 and radius.90/grid4096), not the old
zero-constant finite convolution. Nonzero offspring0 means arbitrarily
high outer powers contribute inside a low-order window, so the old
window proof does not transfer. Require absolute coefficient discrepancy
<=1e-8 and degree1 agreement with the independent chain derivative<=1e-9.
Compare only to existing curve_v2 and n64/d6 verification receipts;
do not regenerate any sampling measurement. Preserve the old package.

This corrects a theory attribution, not the empirical MC/CV/subspace
failure receipts, and does not authorize another quadrature sweep.
Primary reference: Lee et al., Deep Neural Networks as Gaussian Processes,
https://arxiv.org/abs/1711.00165 ; standard raw ReLU kernel also implemented
in https://github.com/google/neural-tangents/blob/main/neural_tangents/_src/stax/elementwise.py .
