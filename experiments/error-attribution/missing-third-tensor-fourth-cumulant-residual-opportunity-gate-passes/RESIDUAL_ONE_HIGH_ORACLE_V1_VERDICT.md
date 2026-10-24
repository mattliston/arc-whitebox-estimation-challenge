# Missing third tensor: q4 residual opportunity gate passes at one cell

September 5, 2026. Phase 2 only. No estimator, score, fit or submission.

Exact dense width8/12 projection and coordinate-star certificates pass
below9e-16. Source-v2 rank35 postL12 parent CP21568 remains unmodified;
its next32 K21 rows match the previous receipt BYTE EXACTLY. The q4 basis
is frozen from the pre-target host covariance (14.8067% covariance trace).

Two NEW independent8M streams, with seeds distinct from each other AND
the archived host/truth seeds, collect only four full post-third slices
and32 next-K21 rows. Centered float32 MPS products, float64 accumulation;
71.28s/75.98s and less than0.95GB host peak each, sequential GPU jobs.
Slice diagonal consistency4.11e-7/4.30e-7, double-slice symmetry5.77e-8/
5.70e-8. Shared float32 arithmetic bias is not independently measured.

For (I-Q^3)(T_true-T_parent), with its coordinate-repeated entries then
removed exactly, the fixed next-covariance-consumer-weighted results are:

- Baseline observed K21 response MSE1.55864301e-9; cross-stream signal
  point estimate1.51909224e-9; sampling-noise fraction2.5375%.
- Signal natural-strength error ratio .315424912; correction/error
  cosine .827756638; correction norm .803140295 times error norm.
- Independent cross-stream ratios .343131581 and .336631922.
- Unweighted signal error ratio .128959464. Keeping projected repeated
  entries too changes weighted ratio only to .309826898.

The fixed <=.7, cosine>=.6, both cross-stream<1 gate PASSES. This says the
missing late-layer third structure has a useful low-dimensional index
sector, despite failure of whole-third replacement. It does not give
deployable labels, an unbiased confidence bound, a recursive carrier,
final-layer MSE or adjusted gain. Do not reopen whole-tensor compression.

Next authority from this gate is only a separate source/representation/
cost design. No additional rank/layer/seed sweep or full training follows
automatically. The immediate design question is whether a much cheaper
vector/covariance-trace residual captures the same information, before
attempting q full-matrix residual dynamics.

Receipts: residual_one_high_oracle_certificate_v1.json,
residual_one_high_inputs_r35_l12_v1.json/.npz,
residual_one_high_truth_r35_l12_A_v1.json/.npz and B equivalents,
residual_one_high_oracle_r35_l12_v1.json/.npz. All jobs completed.
