# Independent Gaussian trajectory: fixed disagreement pilot v1

Written 2026-09-06 02:56 PDT; Phase 2 only. This is not a modification of
the closed matrix-error observer. That observer transported fitted signed
matrix sources through the bank's tangent. This experiment propagates a
separate, nonlinear, unlearned Gaussian mean/covariance trajectory from the
original input. The existing bank and its complete prefix remain frozen.

Hypothesis: disagreement between two independently propagated closure orders
contains useful missing-history information not available to the current
final predictor. Search of the current Astra and G22 records found no prior
independent Gaussian-shadow disagreement pilot. No universal claim is made
about the representational limits of the previous failed families.

## Frozen construction

Start with mean zero and covariance identity. Each layer uses W.T @ mean
in float64 and W32.T @ C32 @ W32 for covariance. Apply exact univariate
Gaussian ReLU moments. Off-diagonal post-covariance uses Gaussian Hermite
orders 1 through 6, with the exact univariate variance on the diagonal.
There are no higher cumulants, learned corrections, active pruning, or
fitted parameters in this shadow. The off-diagonal Gaussian covariance
is truncated, not an exact bivariate Gaussian integral.

The final predictor receives exactly eight extra features:

1. (shadow pre-mean - bank alpha * bank sigma) / bank sigma;
2. log(shadow sigma / bank sigma);
3. (shadow Gaussian post-mean - bank uncorrected host) / bank sigma;
4. shadow gate - bank gate;
5. row mean of off-diagonal normalized covariance disagreement;
6. row RMS of that disagreement;
7. disagreement @ bank alpha / width;
8. disagreement @ (bank host / bank sigma) / width.

Covariance disagreement is (shadow final pre-C - bank tape final pre-C)
divided by bank sigma outer product; its diagonal is explicitly zero.
Bank scalar inputs come from the already byte-qualified actual scalar-final
rows in rows_matrix_error_observer_v1. The off-diagonal bank covariance
comes from the original full-push tape, whose final diagonal arithmetic is
not identical to scalar-final arithmetic; the diagonal is not consumed.
Bank alpha and gate are stored float32 scalar-final features, not claimed
to preserve the unrounded internal float64 values.

## Qualification before targets

Small synthetic SPD tests must agree with existing make_nonlinear(6,2,True,
True), with all third/fourth cumulants set to zero, within 1e-10 absolute.
Input-only width-1024 smoke rows 35 and 100 must be deterministic, finite,
have positive pre-variances, and preserve every saved bank input/output
byte-for-byte. All 24 pilot rows must pass the same checks. At most two
nice-10 workers, each with two BLAS threads. No labels in row generation.

## One matched fit and decision

Ranks 35..58: train on 20, hold out 37,39,42,48. These holdouts have been
used in other research and are not a fresh competition validation panel.
Existing 1e9-sample final means are accessed only after input qualification.
TRAIN-only extra-feature center/std (floor 1e-6), clip standardized inputs
to +/-8. Two matched 134->128->128->1 heads initialize from the current Q;
extra weights start at zero, and control extra inputs remain zero.

Fixed seed 930073, 40 epochs AdamW, lr 5e-4 cosine to 5e-5, weight decay
2e-5, batch 8192, gradient norm cap 5. Final epoch only. No checkpoint or
seed selection. Independent NumPy held-out evaluation must agree with MPS
delta within 2e-9, and retain the bank's final nonnegative clipping.

Advance only if pooled raw source/bank <=0.65, source/control <=0.70,
and at least 3/4 bank wins. This deliberately large raw requirement accounts
for an additional full covariance trajectory. A rough, unmetered extra-cost
expectation is 0.03-0.04 budgets, not a measured score projection.

Failure closes this pilot: no feature/window/order/scale/seed/epoch/capacity
sweep, 200-net expansion, estimator port, or submission. Success requires
a separately designed full cost/runtime qualification and expanded fit,
then both 14-net panels and persistent official-client validation before
any bank. No forum posts, edits of frozen packages, or commits.
