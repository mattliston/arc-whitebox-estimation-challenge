# Causal matrix-memory error observer — fixed pilot

September6,2026, Phase2. A distinct read-only observer, not a covariance
write, more epochs of the closed instantaneous decoder, or a source re-score.
Earlier learned kernels were tested on fresh coherent covariance errors;
the late missing third sector was independently found to be carried-
dominated. Here ask whether a full-matrix causal memory of analytic source
features predicts the ACTUAL frozen bank's final mean error across nets.

The primary CP/star/trace/P/Q estimator remains unchanged. From its frozen
on-policy per-layer pre C, Gaussian alpha/gate/pdf, sigma and30 features,
run an auxiliary linear recurrence for a mean-error vector e and covariance-
error matrix E, initially zero. Define v=sigma^2, mg=Gaussian ReLU mean:

    u = W^T e; V = W^T E W; z=diag(V)
    e_new = g*u + pdf*z/(2*sigma)
    dg = pdf*u/sigma - alpha*pdf*z/(2*v)
    E_new = (g g^T)*V + Cpre*(dg g^T + g dg^T)
    diag(E_new) = (g-mg*pdf/sigma)*z + 2*mg*(1-g)*u

This is explicitly the tangent of the Gaussian gain off-diagonal map with
exact Gaussian marginal diagonals, NOT the derivative of the full cumulant
carrier or GNN. After layers1..14 add a learned symmetric source:

    S_ij = 1e-4*sigma_i*sigma_j*alive_i*alive_j *
           sum_{r=0..2} R_ij^r (F_i^T Theta_r F_j).

F is the existing14 analytic local response features from the earlier
kernel: g,pdf,alpha*pdf,(alpha^2-1)*pdf,host/sigma,alpha*g,
skew times(g,pdf,alpha*pdf,host/sigma), and kurtosis times the same four.
Three symmetric14x14 Theta, shared across source layers. Source at0 is
zero because the first Gaussian ReLU layer is already known. The observer
never changes main-card means, C, CP, features or recurrent state. Only
its final e15 is added to the bank's final reported mean, then clipped.
No covariance PSD claim is needed for an auxiliary signed error matrix.

Local control uses the SAME source parameters/features but only source14
and the immediate variance-to-final-mean response, without earlier memory.
This tests causal matrix memory rather than another capacity increase.
No true covariance labels, new sampling or main-head refitting are used.

Prerequisites: synthetic exact linearity in Theta and numerical tangent
check of the stated Gaussian map (not the full bank), max relative<=1e-6
in float64. NumPy float32 and MPS float32 observer outputs/gradients agree
within1e-4 relative on fixed small data. Current rank35/100 frozen zero
observer preserves the actual scalar-final bank exactly; fixed random
nonzero Theta numerical parity <=1e-4 relative, no labels used. All input
arrays must be the existing rounded on-policy tape fields; actual final
sigma/alpha/pdf come from the bank's scalar-final row rather than the old
full-final mean head. No boundary tolerance or probe-seed sweep.

Only after prerequisites: ranks35..58,20train/held-out37/39/42/48. Existing
on-policy tapes and finalQ inputs/targets only. Normalize F by TRAIN RMS,
floor1e-6. Both arms zero Theta,100 epochs Adam LR.01 cosine to.001,
seed930072, ridge1e-3*mean(Theta^2), global gradient cap5. Per-net updates;
objective actual final-mean MSE normalized by TRAIN bank MSE. Final epoch
only. Three symmetric matrices are the only learned parameters. One MPS
job, at most two low-priority CPU preparation workers; ten-minute fit cap.

To earn a200-net round require memory/bank pooled held-out raw MSE<=.75,
memory/local-control<=.85, and>=3/4 wins against bank. Also report every
holdout separately. Failure closes this fixed recurrence/features/window/
optimizer; no more epochs, parameter/scale/basis/source/window sweeps or
state writes. PASS still needs actual NumPy replay, native metering/parity,
full14-panel and persistent resource checks. No bank before those gates.
