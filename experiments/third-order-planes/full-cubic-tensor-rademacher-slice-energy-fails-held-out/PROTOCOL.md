# Full cubic slice energy — prospective producer and information study

This branch is not a change to the banked estimator. Lazy331599 remains the
incumbent. The existing eight-mode conditional-quadratic tensor feature and
many other tensor/mean regressions failed; this is a different descriptor,
not evidence that those failures have been reversed. The failed LMIO model
and its frozen terminal payload remain closed.

## Fixed mathematical object

Use all signed CP columns/cohorts in the actual pushed parent bank, the
propagated coordinate star with its original active mask and retained cold
diagonal, and the trace tensor from the incoming active covariance and b.
Covariance is cast and symmetrized in FP32; weights, factors, A, b, sigma,
and Rademacher probes are FP32. This explicitly rounded auxiliary symmetric
tensor is not the exact sequence of rounding in the native K21 producer.

For 16 fixed independent Rademacher probe pairs G,H divided by sqrt(1024),
form Z_ij=T_i(G_j/sigma,H_j/sigma)/sigma_i and q_i=mean_j Z_ij^2.
Seed 2026092059 is chosen without reading this branch's labels or vectors.
All CP, star and trace contractions are added BEFORE squaring. This retains
their cross terms. Its expectation over the probe ensemble is the squared
Frobenius norm of the standardized tensor slice divided by 1024^2. A single
fixed 16-probe realization is noisy and is not an exact energy or a confidence
bound. CP signed pairs use an algebraically equivalent center/difference
factorization to reduce cancellation. No seed/count/layer sweep is allowed.

## Producer qualification (before any branch label decoding)

All numerical work runs serially under run_guarded.py: nice10, six BLAS
thread controls1, group RSS below8GB, zero added swap, no GPU. Fixed sources
and metadata are hash sealed before the first unit call.

Four small dense-tensor fixtures include cold diagonals and closely spaced
signed pairs. FP32 contraction relative RMS <=1e-5, energy <=2e-5, and the
independent FP64 formula versus explicit dense expansion <=1e-10. Repeat
outputs and FLOP counts must agree. Probe initialization is charged.

On real parent calls35/100/35, layers8..15 only, compare to independent FP64
direct signed-CP contraction of the same rounded inputs. Relative RMS must
be <=.001 for Z and <=.002 for q at every layer. Record all results, including
failures. Every prediction/feature byte must equal the original parent;
whole FLOPs minus all measured added producer and setup FLOPs must equal
each original cold/warm parent receipt. All new feature bytes must repeat.
Added FLOPs must be <=7.5% of the parent on each call. Research FP64 timing
is separately recorded and is NOT a client runtime qualification.

If qualified, acquire six cold TRAIN captures150/200/250/300/400/500 with
the same gates and parent parity. Save only small features, never every
layer's complete factors. Reference arrays are existing coherent8M samples;
there is no new sampling.35/100 are network-disjoint CHECK versus this fit,
but are extensively reused public research networks, not pristine holdouts.

## Prespecified information screen

After producer qualification, seal learning implementation before decoding
reference arrays. Fit once, no hyperparameter or layer selection. Predict
normalized incoming cumulant errors: third with per-layer alpha and parent
skew controls, fourth with a per-layer constant. Extend the third model by
global q*alpha and q*parent-skew terms and the fourth by global q. Normalize
q by TRAIN per-layer mean and standard deviation. This gives24 control
and27 extended coefficients. Models are linear, separately solved inFP64
with ridge1e-3 after weighted column RMS scaling. Ridge convention and
equations will be recorded in the sealed learning code. Physical weights
are the parent's marginal mean sensitivities times sigma, so combined loss
is the sum of third and fourth mean-response error energies. No parent
neural payload is modified or refitted.

The extended model must reduce pooled CHECK source loss by at least15%
versus zero prediction and5% versus the matched fitted control, with each
network no worse than either baseline. Evaluate all8 layers8..15; no subset
selection. If this screen fails, close the branch without a candidate or
1B target access. If it passes, an actual native implementation must pay
the producer/consumer cost and pass ordinary cold adjusted ratios<=.98 on
both35/100 before the established NC14/client/contract/package release
gates. Information loss is never reported as a predicted official score.

User preauthorization covers qualified future board submissions; no new
permission request is needed. No upload occurs for an unqualified branch.
