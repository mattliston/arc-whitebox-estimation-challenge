# Current-S31 covariance supervision: fixed fit failed

No new score improvement or submission. Both excluded networks miss the
fixed covariance and following variance-response gates.

| Excluded MINI | Covariance residual ratio | Response residual ratio | Joint gate |
|---|---:|---:|---|
| 14 | 0.982267233206 | 1.048420313907 | Fail |
| 71 | 0.961187539259 | 0.950966398241 | Fail |

Each network needed covariance residual <= 0.90 and response residual <= 0.85.
Covariance errors improve modestly, but one response worsens. The arithmetic
mean TRAIN ratios are 0.981410892418 for covariance and
0.878452821871 for response. These are proxy errors, not
final mean MSE or adjusted scores. No final-layer target mean was decoded in
this study. The reference is a single 100M-sample estimate; its noise and
shared numerical bias were not removed.

The new 6,177-parameter pair network was trained on cumulative post-layer-14
covariance error of current S31. Its raw30 + H64 node fields and normalized
C, K21 pair fields match the previous final-mean-trained architecture, but
all parameters were freshly initialized. The correction has zero diagonal
and annihilates the current predicted mean. Sixty epochs / 360 whole-network
updates used six TRAIN networks. Model and all eight predictions were frozen
before reading the two CHECK covariance labels. All eight public MINI
networks have substantial prior research use; they are not pristine holdouts.

Eight baseline calls and eight instrumented calls preserve every output byte
across all 16 layers. The instrumented calls additionally pay for a full-row
CP/star research probe, so their bills are not deployment bills. All reference
identities, sample counts, moment diagonals and affine mean/covariance checks
pass. The proposal consumer's zero correction stays within its fixed rounding
bound on all eight networks. No fitted native candidate or native fidelity,
final-score, bill, VALID40, client, archive or upload qualification is claimed.

Independent NumPy model replay with a dense constrained projection agrees
with Torch to 2.07e-07 relative error.
Twenty-four independent full-width variance contractions agree to
1.83e-19 absolute error.
Scalar summation reproduces the reported scores to
7.77e-16. Synthetic projection and all-parameter
directional-gradient checks pass. Sources, inputs, model, targets and
predictions remain pinned.

This closes the fixed covariance-supervised local pair supplier. It does not
bound every covariance representation or supplier with additional nonlocal
information. No checkpoint, amplitude, regularizer or label rescue follows.

All five guards complete and retire, peak 5.402870 GB, zero added swap.
No new samples or downloads. The saved official S31 #332517 score remains
2.1736465056908115e-9 adjusted; 2e-9 and approximately 1e-9 remain unmet.
