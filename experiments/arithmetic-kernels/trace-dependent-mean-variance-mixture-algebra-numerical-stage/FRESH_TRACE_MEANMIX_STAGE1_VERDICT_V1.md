# Trace-dependent mean/variance mixture: algebra and numerical stage

September 6, 2026. Phase 2 only. No score, value experiment, fit or upload.

This supplies one concrete new raw-accuracy hypothesis: integrate the
correlation between the existing trace vector b and a Gamma variance
scale, then remove the mixture's own changed third/fourth Edgeworth
response. It is not the old zero-mean-shift Gamma consumer reopened.
The exact equations and subtraction ownership are in
FRESH_TRACE_MEANMIX_DESIGN_V1.md.

## Completed evidence

- Exact small third/fourth tensor and K21/K31/K22 identities agree
  within1.92e-13. The v1 float64 raw-to-sixth-cumulant calculation fails
  its1e-11 relative gate through cancellation. It stays failed.
- The separate80-digit Gamma-polynomial/Wick reference passes the
  unchanged third-through-sixth identities, directional relative
  errors below4.63e-73. This validates algebra, not physical accuracy.
- Rounded actual-bank tapes, fixed ranks35/100, layers8..14: all14
  necessary active-space conditions pass with substantial margin,
  b^T C_active^-1 b/q=.4189..4979. This is not an exact PD certificate.
- The exact original coherent-history rank35 replay passes its old
  input and host byte-equality checks. Captured b/B are from the actual
  source-v2 history protocol, not inferred from labels.
- Eight versus32 nodes passes every fixed late cell: relative mean
  disagreement <=1.534e-13, relative covariance <=2.865e-12, and relative
  induced covariance-increment disagreement <=4.978e-8. Gates were
  1e-8,1e-8,1e-3 respectively. No node/strength/range fitting.
- Exact-call stored matrices do NOT provide a global Gaussian model:
  min C=-1.36e-7..-1.81e-7, min D=-3.33e-7..-4.39e-7,
  min V=-2.38e-6..-5.32e-6. C/B were explicitly symmetrized for the
  proposed model, maximum changes1.53e-7. Nothing was PSD-clipped.
  Every queried conditional bivariate Gaussian was numerically valid,
  but that is not a certificate of one positive joint distribution.

Exact worker completed15.11seconds, peak5,207,359,488bytes, nice10 and
three CPU threads. No process remains live. It read fixed coherent
history inputs, not current-output errors for scoring.

## Cost correction and rollout ownership

The implemented information consumer supplies only mean/raw-second
increments. With direct dense elementwise evaluation, sixteen Gaussian
Hermite6 pair-AA maps plus exact-K31 Edgeworth subtraction cost about
12e9 extra FLOPs per16-layer MLP, roughly2.5% of the490e9 bank bill.
This is an analytical operation estimate, not a flopscope meter.

A physically coherent full-state rollout also needs pair-BA and pair-BB
raw moments, and marginal third/fourth raw moments, all centered after
the changed mean. The same component derivatives can be reused, but
these pair products raise a rough complete consumer estimate to30e9
extra FLOPs (~6% bank). Leaving the old K21/radial centering unchanged
would not implement that full model. No such rollout is implemented.

No added dense weight push is inherent in the pair consumer. Production
positive-definiteness verification or a representation repair is not
yet solved or priced. A Cholesky-like check would add O(n^3) work; the
existing indefinite F32 matrices cannot be silently called PSD. Remote
operation count, temporary memory, and wall-time also need a real meter.
Batching the sixteen components reduces RPC count but adds n^2-sized
buffers, so it cannot be assumed free on the bank's memory footprint.

## Next permitted decision

The probability interpretation remains unqualified. Root must decide
whether to investigate a separately stated formal pair closure (valid
bivariate kernels but no positive global distribution), or to repair
the covariance representation with an explicitly priced and validated
change. Neither converts the strict-PD flag to true. No value labels
were scored, and the fixed covariance<=.95 / next-value<=.90 /5-of7
gate has not run. There is no raw or adjusted improvement to bank.

Evidence files:
fresh_trace_meanmix_algebra_v1.json, fresh_trace_meanmix_algebra_v2.json,
fresh_trace_meanmix_feasibility_v1.json,
fresh_trace_meanmix_exact_r35_v1.json/.npz.
Helper SHA d5efe624cd17ac29ebcd714d1ec5d6a8a713c9026c0f7efc0d26d976864155b9.
Pair consumer SHA2e304d3e55632809ec956c0a329ad3fc7bd7e5d0fb989e057a450c1c728bd2af.
