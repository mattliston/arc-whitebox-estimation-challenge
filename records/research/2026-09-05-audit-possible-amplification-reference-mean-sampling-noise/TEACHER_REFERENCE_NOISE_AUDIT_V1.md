# Audit possible amplification of reference-mean sampling noise

September5,2026. Phase2 only. No estimator change or proposed submission.

Several honest teacher-feature probes leave nearly the same~4.8e-9
residual. Before treating that as entirely missing physical information,
check whether noisy reference means fed back through covariance contribute
appreciably to the teacher residual. This is a data-quality sensitivity
audit, NOT reopening sampling/quadrature as an estimator.

The stored organizer-derived truthpost files contain float32 mc_post and
mlp_id, but no sample-count or sampler metadata. Existing source comments
identify them as1e9-sample means. Treat independent1e9 Gaussian Monte Carlo
as an EXPLICIT NOISE MODEL, not a newly verified sampler provenance claim.
First-layer exact-mean errors are consistent in order with that model.

For8 independent fixed-seed Gaussian inputs, obtain each16-layer output
path h^(j). e_l^(j)=(h_l^(j)-stored_mean_l)/sqrt(1e9) has the cross-layer
second moment of an iid1e9 mean's noise (up to tiny centering error).
It is not itself a1e9 CLT draw; the symmetric response uses only its
second-moment role. Evaluate the unchanged teacher-forced full card with
stored means+e and-e, all current features formed before correction,
and record the frozen head's residual. Baseline must be byte-identical.
Use half their residual difference to estimate propagated teacher-label
noise variance, including correlations across layers and GNN inputs.

Ranks35/100 only;8 fixed noise directions each; two nice10 workers with
3BLAS threads. No new training fit, labels, sealed data, sampling estimator,
or actual-card parameter changes. Compare layers10..15 mean squared noise
response with the existing honest frozen-head teacher residual MSE. If
the model estimates<10% on both, close this confounder at that scale; if
>20% on both, verify actual label sampler provenance and response linearity
before contemplating any changed training protocol. Ambiguous results do
not justify claiming that stored truths are bad or noise is the raw floor.
