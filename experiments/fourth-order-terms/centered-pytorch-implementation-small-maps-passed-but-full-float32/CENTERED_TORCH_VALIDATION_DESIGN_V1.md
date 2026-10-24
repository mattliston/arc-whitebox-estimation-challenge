# Centered Torch numerical validation, fixed gates

September6,2026. Continue CENTERED_MOMENT_NUMERICS_DESIGN_V1.md, whose
scalar/pair/full NumPy certificates passed. No changed physics, targets,
training, payload, deployment candidate, bank or score claim.

New centered_torch_numerics_v1.py is a literal differentiable translation,
including torch.where reflection at zero and the original asymmetric
covariance convention. centered_bank_autodiff_v1.py compiles copies of
the original prefix/final step bodies into private globals; it replaces
only the CDF, scalar/pair maps and birth beta, and adds optional selector
auditing. Scalar finalQ is included. Do not mutate the original modules.

Small fixed validation precedes wide runs. Reuse the66 scalar cells from
the120-digit NumPy certificate. CPUf64 forward sigma-normalized absolute
errors must be<=1e-10; CPUf32<=2e-5. Compare scalar derivatives with the
original f64 raw formulas, including all six zero-mean cells and all four
normalized inputs (alpha, log sigma, standardized third/fourth cumulant).
Require maximum normalized derivative error<=1e-9 in f64 and<=2e-4 in f32.
Use pair widths11/17, covariance asymmetry0/1e-6, seed930092. Pair forward
f64<=1e-10 and f32<=2e-5. Compare reverse gradients of four fixed weighted
output contractions against the original f64 pair map, with aggregate
relative tolerance1e-9/f64 and2e-4/f32. No tolerance change after results.

MPS erfc must work with PYTORCH_ENABLE_MPS_FALLBACK=0. Test only when no
external Python compute job is observed. No silent CPU transfer. The
fullf32 bank on BOTH35/100 must pass the original2e-6 maximum mean gate
against the frozen NumPy baseline, with all30 active/birth membership
arrays equal. Record order separately. CPUf32 can run first. Full f64
Torch parity is a diagnostic control if f32 fails, not a new candidate.

Only after small derivative and full forward parity may full local
finite-difference/autodiff cell checks proceed; no training is authorized
by numerical parity alone. A new training hypothesis needs its own
preregistered lever and must respect previously closed experiment families.
Run AP Torch without threadpoolctl, threeCPU threads, nice10; one MPS job.
