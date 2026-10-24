# Three-leg covariance completion: closed at representation gate

Phase2, September5,2026,22:50PDT. No labels, fit, rollout or submission.

Independent dense n6/n9 tensor, plane, push, adjoint and star-restoration
identities pass. Iterative inverse certificates reach<1e-10 residual.
The full-width current bank covariance is SPD: standardized live-block
eigenvalues .01457538..31.44265 (832 active), so no jitter was needed.

At rank35 post12, replacing the21568-column CP component with the
predeclared two-PCG covariance-metric completion and EXACT repeated-plane
star restoration gives weighted next-W13 K21 error .3378142155 relative
to the original complete CP+star+trace tensor (gate .03 FAIL). Unweighted
error .3742814450. The two-step linear residual is .0406739940.

The30-step diagnostic reaches3.85548e-7 linear residual, NOT its1e-10
convergence criterion, and still has .3167027527 weighted next-plane error.
Do not call it an exact converged projection. Both arms' post repeated
entries are restored to<1.46e-17 relative, including the practical arm.
The current repeated plane alone does not supply the missing distinct
structure in this representation. No iteration/metric/regularizer sweep
or cost/rollout expansion is justified by this fixed test.

5.405s, peakRSS4,640,669,696bytes, one nice10 four-thread CPU process.
Receipts: covariance_metric_completion_certificate_v1.json and
covariance_metric_completion_wide_r35_l12_v1.json/.npz. No board score.
