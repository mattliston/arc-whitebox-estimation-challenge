# Fixed dominant-direction squared-fluctuation regression information test

September 5, 2026. The joint moment PSD condition holds; no repair is
indicated. It motivates a distinct, target-blind observable rather than
a correction chosen from the failed PSD hypothesis.

For the top eight eigenpairs (u_a,lambda_a) of pre-covariance C, define

    b_ia = Cov((Z_i^2-v_i)/v_i, u_a.Z/sqrt(lambda_a))
         = sum_j T21_ij * sigma_j*u_ja/sqrt(lambda_a),
    c_ia = Cov(Z_i/sigma_i, u_a.Z/sqrt(lambda_a)).

The four basis-rotation/sign-invariant observables are sum_a b_ia^2,
sum_a b_ia*c_ia, sum_a c_ia^2, and
sum_a (b_ia-skew_i*c_ia)^2. The first is conditional linear regression
energy of the squared fluctuation; the last removes a purely marginal
skew response. These are not the old covariance loading alone, fixed
response-weighted plane contractions, or the mean/two-direction cubic
features. Eight modes are fixed before any result; no rank/basis sweep.

Reuse the exact saved teacher-forced matrices/h126/residuals for ranks
35..58, layers10..15, from the leakage-audited joint-graph pilot. Restore
the correlation diagonal to one and the third-plane diagonal from the
parent's pre-skew feature. Recover physical C using the saved sigmas.
All features depend only on current pre-state, before current correction.
No new MLP executions or new Monte Carlo samples. No sealed targets.

Train two same-capacity 130->64->64->1 SiLU residual heads from identical
initial weights, zero output: one receives h126 plus these four TRAIN-only
RMS-normalized observables, one receives h126 plus four zeros. Same fixed
20train/4heldout(37/39/42/48), seed5290723,40epochs, AdamW1e-3 cosine1e-4,
weight_decay2e-5, gradient norm cap5. Output scale sigma*parent_layer_rms.
Use the final epoch, not best epoch.

Information gate: held-out mean residual MSE <=.92 of same-capacity control,
<=.95 of parent and improvement on at least3/4 held-out nets. Failure closes
this fixed conditional-energy feature set without more rows, rank sweeps,
200-net coadaptation or port. Passing would still require current-carrier
closed-loop, scalar-final contraction, cost and 14-panel validation. No
information-test number is a candidate score.
