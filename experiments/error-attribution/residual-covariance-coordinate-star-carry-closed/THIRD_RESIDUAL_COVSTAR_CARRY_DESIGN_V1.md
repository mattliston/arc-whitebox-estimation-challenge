# Can a repeated-plane residual carry useful joint information?

September6,2026, Phase2. Fixed public mini57, pre13 -> pre14 -> mean15.
Use only the admitted first training100M file and unchanged bank. The
closed coordinate-covariance-star prototype replaced the WHOLE third
tensor. Here retain the whole parent and complete only its known oracle
off-diagonal K21 ERROR at pre13. This is a residual-carry observability
test, not a new coefficient fit or a reopening of whole-state compression.

Standardize with actual pre13 sigma and correlation R. Remove the residual
diagonal. Set A_ii=0 and solve the exact two-by-two plane equations:

    S_ij = 2 (A_ji + R_ij A_ij),
    A_ij = (S_ji - R_ij S_ij)/(2(1-R_ij^2)).

The tensor is the six-term symmetrization of columns
(sigma*R, diag(sigma), sigma*A). This matches every off-diagonal S entry
and has zero scalar third marginal. Use symmetric actual R, no clipping,
fit, ridge, damping or selection; require max off-diagonal |R|<.98.
Transform all three legs by J=diag(actual Gaussian gate13)*W14.
Read its full next K21 plane exactly, then discard the next diagonal too.
This isolates joint third-error carry through the linear gate term; fresh
nonlinear/covariance-error sources are intentionally not supplied.

Certify plane matching and affine push against small explicit tensors,
including row sigma scales, nonsymmetric S and varied gates, <=1e-10.
Before labels are used at the next cell, freeze actual pre14 K21/C/m/aux
and actual scalar-final pre15 variance-to-mean response. Replay must
reproduce all original frozen pre13 input/prefix arrays and the previous
pre14 response byte-for-byte. No current truth enters that trajectory.

Admit pre14 M21/M11/marginal fields from the already downloaded training
file. Same affine-consistency and stored-f32 rounding bound checks as the
qualified pre13 cell, with unchanged tolerances. No held-out file, new
sampling or official1e9 target. Target is actual true pre14 off-diagonal
K21 minus actual bank K21. Evaluate natural predicted increment through
the exact third-only covariance response and its fixed W15 Gaussian
variance-to-mean consumer. Require BOTH error ratios<=.75; report weighted
plane fidelity too. These are ordinary single-stream oracle errors, not
noise-separated final MSE or adjusted score.

Failure closes this fixed residual completion/linear-carry route without
changing rank, cell, frame, scale, tensor completion or response. Success
would only support a separate source-identification/recursive-state
design; true input K21 residual is unavailable in deployment. A possible
implementation needs two general factor pushes and three plane products
per transition; no complete bill or float32 fidelity has been established.
No learner, candidate, panel, forum post or submission follows directly.
