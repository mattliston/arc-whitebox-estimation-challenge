# Current covariance negative-mode information test

Phase 2 only. Selected 2026-09-21 UTC before this study's numerical work.
Exact official331684 parent facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0.
The bank remains adjusted3.5844144212297072e-9,100/100 complete.

Hypothesis: a small set of invalid negative covariance eigenmodes may be an
actionable source of error. Earlier diagonal-preserving full spectral repair
was included in a much older high-order architecture with a conservative
330.683GF repair allowance. Neither that design nor the failed current
symmetry-only candidate measures the present negative eigenspace. This is
a distinct information/cost-feasibility question, not a claim that enforcing
positive semidefiniteness supplies missing higher cumulants.

Reuse authenticated unchanged-native captures from
full_lower_covariance_innovation_supervision_v1/qualification on35/100.
No new forward call or copied dataset. For every post layer8..14, let
S=(C+C.T)/2 in FP64. Diagonalize S offline. From its ascending eigenvalues
take at most the first16 negative eigenmodes, without a threshold, rescaling,
ridge, layer selection or learned coefficient. Set

    D = Q diag(max(-lambda,0)) Q.T.

The proposed action is C+D, retaining the parent's original antisymmetric
part and permitting diagonal changes. It is not a nearest-correlation
projection, the previous symmetrization candidate or guaranteed full PSD
repair if a negative tail remains. The full negative spectral part of S
is a standard projection construction; the particular limited action and
consumer test here are our fixed choice. Primary reference:
https://eprints.maths.manchester.ac.uk/363/ (Higham,1988).

First certify small explicit matrices, permutation/orthogonal covariance,
zero/PSD/indefinite cases, dense versus factored responses, and Gaussian
variance-to-mean finite differences. Tolerances1e-10 algebra,1e-7 derivatives.
Wide eigensystem residual/orthogonality<=1e-10. Record every eigenvalue,
negative count/energy, top16 energy fraction, and correction norm/trace.
The rank16 action must retain >=99% of negative squared-eigenvalue energy
at every one of14 cells. A zero negative energy gives fraction1. Failure
closes this fixed low-rank approach before target decoding; no rank rescue.
All factors and geometry records are sealed before labels.

Only if the spectrum gate passes, open the existing coherent8M covariance
references35/100 and compare E=C_true-C to E-D. Also compare the immediate
next Gaussian variance-to-mean error y to y-z, where

    y_i = pdf_next_i/(2 sigma_next_i) * (W_next.T E W_next)_ii,
    z_i = pdf_next_i/(2 sigma_next_i) * sum_a d_a (Q.T W_next)_ai^2.

All14 layers contribute without weights or selection. EACH network must
have pooled covariance squared-error ratio<=.98 AND next-mean-response
ratio<=.95. These are frozen-state proxy errors on reused research networks
with8M sampling noise, not noise-corrected, independent final MSE or adjusted
scores. No ordinary1B target is opened. No coefficient fitting occurs.

Failure closes this exact negative-mode action: no diagonal normalization,
rank, threshold, gain, layer, label or fit rescue. A pass only licenses a
separately fixed native eigenspace approximation and actual complete-cost,
trajectory and final-objective test. No cheap eigenmode acquisition is
assumed here; offline full diagonalization is NOT a deployment cost. There
is no candidate or upload from this screen.

One nice10 numerical group, six BLAS controls1,8GB RSS hard guard,
256MiB emergency swap cutoff, ZERO added swap required. Unit60s,spectrum120s,
score90s,audit90s. New artifacts<=20MB. No GPU,new samples or downloads.
Preserve all previous source/fit/compression/FMM/sampling negatives.

Source review this turn also revisited the finite mixed-FMM envelope,
full signed heavy source, full-star persistence, conditional Gaussian
residuals, third-plane endpoint fit and cubic stochastic noise. Their old
negative scopes remain intact. A bounded public search returned known
Phase2 write-ups, with no newly tested supplier identified in those results.
