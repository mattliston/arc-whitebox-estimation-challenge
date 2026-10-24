# Current-native covariance-innovation supervision

Phase2 only. Selected before this study's new captures, target reads or fit.
The exact official331684 parent is facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0.
Official adjusted3.5844144212297072e-9 remains unchanged;3/2/~1e-9 remain
unmet. Qualified submissions remain authorized. This is an information study,
not a candidate release or a claim that the fitted proxy is the final loss.

The old edge model used oracle previous mean/covariance and older physics.
The old matrix observer fitted final means, not covariance innovations.
The recent native edge-head SPSA and response/precision studies remain failed.
This experiment does not use, assert or qualify their full-trajectory
gradients. It tests a different supervision signal on frozen actual current
states. No extension of those fits, derivative threshold or precision change.

Retain the existing frozen edge encoder and its original sixteen output
weights. At layers8..14, form the sixteen real-linear output features B_r
from its actual FP32 normalized F,R,S, parameters, tanh activations and
rounded sigma/pdf/gate inputs. Widen those stored quantities for the
symmetric output-feature assembly. B_r is the coefficient-r contribution
to the edge covariance source; it includes the original cov_factor.
Fit one shared sixteen-vector DELTA added to the original head. No new
input, hidden weight, per-layer coefficient or physical state is trained.
FP64 linear assembly is a research feature definition; it need not be
byte-identical to a changed FP32 native head. Qualify its base reconstruction
against the actual original edge output before opening labels.

Passive source taps retain exact parent means, post covariance7..14,
pre covariance8..14 and Gaussian fields7..15 plus edge inputs/output8..14.
The exact repeated35 capture keeps checksums and reuses its first archive.
No new arithmetic
is inserted. Prove whole-function source inverses. Qualify full35/100/35
against archived exact-parent all16 bytes and complete cold/warm bills,
including all captured bytes on repeat and numeric-cache retirement.
Then acquire150/200/250/300/400/500 serially, one cold process each. Source
inverse and unchanged operation order remain required; no claim of an
independent original replay on these six new captures. Seal all captures
before any target arrays. At most2GB new data and8GB process-group RSS.

Use existing coherent8M mean/covariance files for TRAIN35/100/150/200/250/300;
VALID400/500 is unseen by this fit, but has prior research history. These
are reused public data, not pristine holdouts or noise-corrected truth.
For each layer8..14, e=truth_mean-parent_mean and
E=sym(truth_cov-parent_cov). With the frozen actual Gaussian fields and W,
predict incoming covariance error by the already audited Gaussian-gain
tangent:

    u=W.T e_previous; V=W.T E_previous W; z=diag(V)
    dg=pdf*u/sigma-alpha*pdf*z/(2*sigma^2)
    T=(g outer g)*V+Cpre*(dg outer g+g outer dg)
    diag(T)=(g-mg*pdf/sigma)*z+2*mg*(1-g)*u
    U=E_current-sym(T).

U is a specified approximate local innovation; it is not the derivative
of the full cumulant/GNN program. Both previous and current labels use the
same8M stream. Preserve and report source affine checks<=1e-4. Compute
the next Gaussian mean response L(U)=pdf_next/(2*sigma_next) times
diag(W_next.T U W_next), using actual parent next fields, including scalar15.

One closed-form shared ridge fit minimizes the sum of mean covariance and
next-response squared errors, each normalized by its TRAIN baseline energy.
Average uniformly over six TRAIN networks and seven layers. Normalize each
of the sixteen columns by the square root of its combined TRAIN Gram
diagonal, floored1e-20; ridge1e-3 times identity in normalized coordinates.
No intercept, feature removal, coefficient clipping, optimizer, checkpoint,
penalty/layer/label/window/roster search or post-result fit extension.
Round DELTA toFP32 once before scoring and save both forms. Use fixed
linear features for the proxy; no changed native trajectory is scored here.

Small independent dense-feature/linear-head/symmetry/permutation and tangent
finite-difference units precede captures; FP64 algebra<=2e-10 scaled,
tangent central difference<=2e-7 relative,
native-edge reconstruction<=3e-6 scaled. Current wide reconstruction must
also pass before targets. Solve residual<=1e-9. Independently verify the
quadratic sufficient-statistic loss against explicit tiny predictions.

Advance only if EACH VALID network has covariance-innovation residual
energy<=.98 of baseline AND next-mean-response residual energy<=.95 of
baseline, pooled across its seven layers. TRAIN and every layer remain
diagnostic. A pass licenses a separately frozen native finite-endpoint test
with the one payload, its own complete bill, and unchanged36/38 development
gate; it does not establish native gradients or permit immediate upload.
Failure closes this exact current-input/supervision/frozen-encoder fit;
no hidden-feature, coefficient, regularization or label rescue in this study.

Serial nice10, six BLAS variables1, no GPU. Existing guard8GB RSS,
256MiB emergency swap cutoff; successful stages require ZERO added swap.
Unit90s; qualification triple240s; each capture120s; wide qualifier180s;
each sufficient-statistics process240s; fit60s. One numerical process group
at a time. No samples/downloads/ordinary1B target access or external writes.
Preserve all scripts, frozen protocols, failed receipts and prior banks.
