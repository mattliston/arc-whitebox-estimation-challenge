# Centered moment rewrite: scalar, pair and full-depth f64 fidelity passed

September6,2026. Numerical re-expression of the SAME moment card; no new
physics, learned predictor, f32/MPS result, gradient, raw/adjusted score,
bank edit or submission.

Scalar66-cell120-digit certificate (06:29:10PDT): standardized means
[-12,-8,-4,-2,-.5,0,.5,2,4,8,12], sigmas .3/1.7, three fixed(k3,k4) cases.
Maximum sigma-normalized absolute errors: mean2.115e-15, variance1.880e-16,
third cumulant2.410e-16, fourth cumulant7.277e-16. Uses reflected negative
Gaussian tail central moments and exact centering of Edgeworth raw
increments, including all powers of the small mean shift.

Pair-map certificate (06:32:41PDT) passes for widths11/17, symmetric and
deliberately1e-6-asymmetric covariance. Maximum normalized K21 error3.09e-14,
radial error9.08e-15. Crucial asymmetry term is retained: the old map uses
p21.T, which is NOT exactly pair(A,B) when f32 Cpre is asymmetric. Direct
small antisymmetric powers and K22 weights reproduce that difference;
the bank covariance is not silently symmetrized.

Full1024x16 NumPy bank replay (06:35:36PDT), current P0 and actual scalar
finalQ, three CPU threads, no labels:

| Development net | Max mean difference | Final mean difference RMS | Final head-delta max |
|---|---:|---:|---:|
|35|1.81815089e-7|3.20575814e-8|8.25821189e-10|
|100|1.86008903e-7|3.60893611e-8|1.24407507e-9|

All30 active/birth membership arrays AND ordered indices match exactly
on BOTH nets. Full reruns use changed centered pair/scalar/birth-beta
arithmetic, not a fixed precomputed trajectory. The small difference is
amplified f32 carrier roundoff, below the fixed2e-6 bank-fidelity gate.
Final normalized feature h max differs4.15444e-5/3.64780e-5; do not claim
those features or whole predictions are byte-identical.

Time25.313s, peak RSS5,627,199,488bytes. Receipt
centered_parent_numerics_v1.json, output SHA256
b77ec029874a66573581ee8118565714f5ea4d06878a1cbd42cdfe269f14ff77.

Next work is a versioned Torch port and explicit f32 numerical/gradient
validation, not training. AP Torch must never import threadpoolctl. The
absolute-value reflection used in NumPy must become a differentiable
piecewise torch.where(alpha>=0,alpha,-alpha), NOT torch.abs at alpha0:
abs has zero derivative there although the original ReLU Gaussian
cumulants do not. Check derivatives at zero against the raw f64 formula.
erfc support on MPS must be tested; no silent CPU fallback may masquerade
as GPU fidelity. The old failed MPS baseline stays preserved.
