# Deep third-cumulant supply deficit identified as a symmetric collective rank-one tensor

2026-08-30 · experiment · informational · measured

THE SUB-RESOLUTION GENERATOR IS IDENTIFIED (18:20, probe-tensor instrument, 8M samples + tensor-side projections, 105 probe pairs, 73 s wall): the deep supply deficit is a FULLY SYMMETRIC rank-1 tensor deficit in the ACTIVE-UNIT (m) DIRECTION - D = Se-St has rel 0.33 on every m-family probe pair (mode_col/err_v1/k3/g - all mutually 0.9+ correlated) vs 0.10 on random controls, k-profile corr -0.977 with m: the estimate UNDERSHOOTS collective-mode third-cumulant content (kappa3 of s = m-hat . dz) by ~1/3, while corr(Se,St)=0.989 everywhere else. The plane frame saw only its (a,a,b) shadow (m^2 x m ~ the k4xm mode - all unit-activity powers correlate deep). MECHANISM: per-pair Mehler bond truncation is pointwise fine but sums COHERENTLY over N^2 pairs along the collective mode; the collective coordinate is a rho=1 coincident-bond object - exactly [[unit-bond-coincidence-law]]'s 'exact resum is right physics' territory, now at w1024 with a measured 1/3 magnitude.

Recorded measurement:

    THE SUB-RESOLUTION GENERATOR IS IDENTIFIED (18:20, probe-tensor instrument, 8M samples + tensor-side projections, 105 probe pairs, 73 s wall): the deep supply deficit is a FULLY SYMMETRIC rank-1 tensor deficit in the ACTIVE-UNIT (m) DIRECTION - D = Se-St has rel 0.33 on every m-family probe pair (mode_col/err_v1/k3/g - all mutually 0.9+ correlated) vs 0.10 on random controls, k-profile corr -0.977 with m: the estimate UNDERSHOOTS collective-mode third-cumulant content (kappa3 of s = m-hat . dz) by ~1/3, while corr(Se,St)=0.989 everywhere else. The plane frame saw only its (a,a,b) shadow (m^2 x m ~ the k4xm mode - all unit-activity powers correlate deep). MECHANISM: per-pair Mehler bond truncation is pointwise fine but sums COHERENTLY over N^2 pairs along the collective mode; the collective coordinate is a rho=1 coincident-bond object - exactly [[unit-bond-coincidence-law]]'s 'exact resum is right physics' territory, now at w1024 with a measured 1/3 magnitude. THE FIX (derivable, billable): exact 1D-conditional recursion for s (condition on s: z|s Gaussian; kappa3(s') via ~40-pt quadrature x per-unit closed forms ~ 4e4 flops/layer) injected as a single w x w x w correction triad. Validation next: does the 1D-exact kappa3(s') reproduce the probe truth's collective value at L12 given the L11 state?

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g19_attribution_v1/mc_probe_tensor.py](code/public_g19_attribution_v1/mc_probe_tensor.py)
- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [mc_probe_tensor.py#L1](code/public_g19_attribution_v1/mc_probe_tensor.py#L1)

[Research graph](../../../graph/index.html#node=2026-08-30-deep-third-cumulant-supply-deficit-identified-symmetric-collective-rank) · [Copy provenance](../../../PROVENANCE.json)

