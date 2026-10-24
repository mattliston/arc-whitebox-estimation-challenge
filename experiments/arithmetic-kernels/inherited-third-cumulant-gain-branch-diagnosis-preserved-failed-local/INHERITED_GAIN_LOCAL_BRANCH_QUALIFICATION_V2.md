# Inherited gain: diagnose the branch, preserve the failed v1

2026-09-05, 22:36 PDT, Phase 2 only. No new accuracy result or submission.

The v1 full-dynamic halving prerequisite failed on rank 100 and remains
failed. Its original training script is not run or edited. The target-free
cause audit exactly replays all three original NumPy predictions. One
active membership swap (two changed bits, unchanged count 832) at layer 12
accounts for the discontinuity. Holding the baseline membership and order
fixed reduces final halving discrepancy from 30.0005% to 1.04947%.

This is the same already-documented phenomenon as the full-carrier,
conditional-birth and coherent-scale certificates earlier today. A failed
finite-step global-linearity test is not an accuracy failure and is not a
test of the local-branch derivative. Treating it as one would contradict
the established differentiation protocol. This follow-up is explicit,
target-free, and does not replace or relabel the original failed flag.

First, use the ORIGINAL theta and half-theta, no new amplitude/direction,
and independently compute the zero-point CPU forward-mode derivative.
Compare it with the two saved NumPy secants: rank 35's unchanged-membership
dynamic path, and rank 100's fixed baseline membership AND order path.
Require final vector-tangent discrepancy <=5% for BOTH amplitudes on BOTH
nets, and preserve the baseline <=2e-6 implementation agreement gate.
This certificate loads no truths and makes no accuracy claim.

Only if that separate certificate passes may a newly versioned pilot use
the original single train-gradient direction, train-only step clipped to
[0,.02], train35/36/38/40, holdouts37/39/42/48, frozen P/Q, 10% pooled raw
improvement and 3/4 wins. No amplitude, layer, radius, optimizer or direction
sweep. All actual forward evaluations retain DYNAMIC selections. The
original failed v1 is still failed. This is a local derivative protocol,
not a differentiable surrogate estimator with fixed deployment supports.

One nice10 CPU process, 3 threads. AP Torch never imports threadpoolctl.
No candidate without all existing numerical, panel and official-runtime
release checks. User <1e-9 goal and no-forum-post rule remain unchanged.
