# Current-P quadratic-response covariance action: scientific gate failed

September 8, 2026 PDT. Phase 2 only. The fixed rank35/post12, amplitude1,
no-fit witness completed at 2026-09-08T14:58:53.185205Z (07:58:53 PDT).
All older sources, failed qualifications and successful certificates remain
unchanged. No native carrier, training, panel, package or upload follows.

## Actual result

The declared current409f baseline uses the exact captured post-P mean
centering order and alive mask, retaining every diagonal. The intervention
adds the qualified covariance response of

    q_i(y)=a_i(y-mu_i)+b_i*((y-mu_i)^2-sigma_i^2),
    a_i=partial_mu deltaP_i, b_i=.5*partial_mu^2 deltaP_i.

Only the P-correction derivatives enter, not the host response or a constant
mean shift. Coefficients and reference center remain fixed inside the
Gaussian/Edgeworth moment operation. This is a declared finite-jet
covariance action, not an exact Gaussian heat semigroup or positive joint law.

The frozen next-mean consumer is J(X)=omega*diag(W13.T X W13), with
omega=captured_pdf13/(2*captured_sigma13). It excludes native rounding,
next-head feedback and all later layers. Predictions were sealed before
opening either reused 8M covariance target.

| Fixed metric | Baseline | Corrected | Corrected / baseline |
| --- | ---: | ---: | ---: |
| Next-mean cross-stream risk |2.4633411085508702e-9|2.455915481776116e-9|.9969855466833326|
| Stream A next-mean squared error |2.953289721514421e-9|2.9455285372230347e-9|.9973720206876938|
| Stream B next-mean squared error |2.9525055822168254e-9|2.9454155129587035e-9|.9975986262986848|
| Secondary covariance cross-risk |1.3617925504782897e-8|1.3673111843420251e-8|1.004052477641912|

The primary gate required cross-risk ratio <=.75 and each individual
stream ratio <=.85. All fail: the primary improvement is only .301445%.
This is not enough evidence to pay for a rollout. No strength, curvature,
layer, threshold or training rescue is admitted for this fixed action.

All 47 numerical/frame/provenance guards passed; error=null. Therefore
this is a scientific failure, not a failed numerical implementation to
repair and rerun. Wall1.09019483346s, peakRSS622886912B; one nice10 OFFPY
worker, three BLAS launch variables1, no concurrent numerical/MPS job.
Empty threadpool inventory is environment-only attestation, not observed
Accelerate thread-count evidence. The process exited1 as designed.

aRMS=9.77106523882197e-5; (2*b*sigma)RMS=.00020218955011109857;
DeltaC RMS=2.0856442600687257e-6; J(DeltaC)RMS=1.267577236072198e-6.
Alive count832. Signal energy1.6067520494084327e-12, signal/error products
A4.683968170397248e-12 and B4.348410653765133e-12. Half-stream
disagreement energy4.895565433147531e-10. These are local reused-development
diagnostics, not final raw MSE or adjusted scores.

## Qualification receipts do not replace the failed value test

The original FP64-wide Q feature-value failure and R coarse-FD feature21
failure remain failures. Independent directed-Decimal interval analysis S
identified the latter as coarse finite-difference truncation error. The
separate prospective T successor used the same interval certificate
uniformly for feature21 first/second derivatives at every fixed FD site;
all other original checks and native-byte comparisons remained unchanged.
It passed all14 current contexts. This does not retroactively pass R.

The T fixed gate-information test passed its point thresholds on both
development nets: learned/host cross-MSE ratios .866590867724 and
.87939647254, equally averaging layers8..14. Margins beyond the .9
threshold were only about1.04/.52 paired jackknife standard errors. This
is neither strong-confidence10% evidence nor final raw improvement.

The separate tiny covariance algebra certificate passed173 fixed checks:
dense/sparse pair formulas, isolated cumulant terms, independent interval
scalar moment integrals, exact null/symmetry and domain/immutability guards.
Those implementation identities licensed this witness, not a rollout.

## Target limitations and state ownership

Both archived streams passed scalar rank35/N8M and every L1..15 affine
mean identity against canonical weights. Stored FP32 covariances were
converted to FP64 and multiplied by8000000/7999999. This removes ideal
finite-N sample-centering bias, not shared MPS-FP32 or storage bias.
No contemporaneous original-A generation/source/seed/weight seal was found.
No covariance batch jackknife or confidence interval exists. The two
references are reused development data, not fresh holdouts.

A future covariance-only intervention would also affect stored trace B,
innovation diagonal, newborn coefficients, subsequent masks/ranks, P
features and all downstream bills. Immediate host/P means and repeated
targets being fixed does not make those consumers invariant. The source
ownership audit is retained as information only; this failed action does
not authorize implementing that recurrence.

## Immutable bindings

| Artifact | SHA256 |
| --- | --- |
| PROTOCOL.md |aed9a97cdf4a297f61b9500e12c3f29b5a7aed37f69b9eb28ba707efee1adaca|
| check.py |b6b94734dae577e3049edd421131c04e51201bf21534602863ffe9cc7bc54e44|
| witness/result.json |e41ec3f2f1f5535568dd5f3cb62c4ac83cb6f2407d78ef16ccd4fb7ee7e86c9f|
| witness/prediction.npz |cdb8668301f893ad6d01ca57cff3aa5363955d1d6b0361f7a38ec60b4ff742a6|
| S/diagnosis/result.json |14b3dbabcd594587f9898acb1f8e3287932e00e9e25bf17347ff5c14a2d92227|
| T/qualification/result.json |52938606d3f116f5f41021fd19f5f3b03f8b9d7c2713741d0d19cbada4221494|
| T/gate_information/result.json |b549ad49f54d134d4110dcc2c476e2f892b2e3018fb6001958bc50750a7f1f07|
| ALG/certificate/result.json |01802bd2032803b88028f8bbe85351a0f4fd66c1fb6fda1e9b2eac14aacd3792|

S/T/ALG are sibling current_response_scalar_certificate_v1,
current_response_certified_native_v1 and current_response_covariance_algebra_v1.
All include their own prospective admissions and original checks.

## Score and next decision

Confirmed official bank #329926 remains4.982231460183304e-9 adjusted.
The best qualified LOCAL409f component remains4.770235434670105e-9 on
independent14, NCraw2.2458484596872894e-8 and mean complete
466058731667.9286F. It is preserved and packaged, not submitted above
the milestone threshold. No complete bill or final raw gain was produced
by this response action; a .9-1.5B/head source arithmetic estimate was
never a metered bound and cannot be combined with the proxy as a score.

The user's immediate priority is official adjusted strictly below3e-9
ASAP; final strictly below1e-9 and qualified banks below4/3/2e-9 remain.
Continue only credible new routes, preserving small qualified components.
Do not force a numerical run when the causal supplier or complete-score
premise is missing. No forum post, commit, promotion or frozen edit.
