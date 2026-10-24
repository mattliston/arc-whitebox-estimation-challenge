# Correlated terminal message energy: fixed information test

The current head transports sixteen linear message channels through W and
uses sum_i W_ij^2 src_ik^2 as their energy summaries. That omits correlations
between incoming units. The new information is a quadratic form of each
message channel against the incoming covariance, approximated in eight
directions. This is distinct from polynomial readout of existing H15.
It is a predictive feature, not an asserted exact cumulant or Gaussian law.

Parent: Cost75 SHA5f66f94afdd2909584527a299c3989377d374289b8f5e1f379047b4176c7521c.
Do not change its source, predictions, or pending submission archive. Observe
the actual C entering final linear transport and actual state entering
gnn_step15, plus its resulting H15, sigma15 and pdf15. Keep the parent call
unchanged and verify cold35/repeat against its existing exact outputs/bills.

Define a formal correlation matrix by symmetrizing C, zeroing its diagonal,
dividing by sqrt(max(diag(C),1e-10)) on both axes, and setting unit diagonal.
The variance floor only defines feature units; no PSD/Gaussian claim follows.
Acquire Q by exactly two QR range passes: Q0=qr(R W15[:,:8]),
Q=qr(R Q0). Form B=Q.T R Q, explicitly symmetric. This rank8 rule is fixed
before examining new labels; no eigen/rank/power/seed search.

For the first16 message channels src=state14 msg_w[:16].T+msg_b[:16],
and each receiving column w_j, compute

    e0_jk = sum_i w_ij^2 src_ik^2
    u_jrk = sum_i w_ij Q_ir src_ik
    eoff_jk = u_jk.T B u_jk
                - sum_i w_ij^2 src_ik^2 (Q B Q.T)_ii
    feature_jk = 0.5 pdf15_j * clip(eoff_jk/max(e0_jk,1e-12), -8,8).

All feature arithmetic uses public flopscope operations in a separate paid
context, including casts, QR, matrix products, scaling and materialization.
Require full component bill <1BF. There is no unmetered eigendecomposition.
Source construction, parent call, exports and work cost are kept separate.
Deployment overhead is not called a complete candidate bill until integrated.

Fixed TRAIN order35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58.
These networks were seen by the base head. This is exploratory network-CV,
not independent validation. Capture all twenty before opening any target.
Store only parent outputs, H15, sigma15,16 new features and scalar receipts
(small files, no full covariance histories). All weights/outputs are hashed.

Five fixed network folds, ordinal modulo5. Compare unchanged parent, affine
H15 readout control, and that same control plus16 new features. Each arm
has an intercept, feature-column train-only RMS normalization, and ridge
1e-4 times the total sample weight. Fit raw mean residual after multiplying
features by sigma15; include each parent's own cold utilization as the
sample weight. Scores use original1B FP32 means widened toFP64, with an
additional measured feature bill for the extended arm and a conservative
1MF/2MF readout allowance for control/extended. Seal all OOF predictions
before aggregate scoring. No fold/ridge/feature/normalization search.

Advance only if extended pooled adjusted error is >=1% below unchanged,
strictly below affine, and below both on at least4/5 fold means. Otherwise
close this feature set without a larger fit or CHECK/native expansion.
Positive information still requires a deployable full20 fit, isolated CHECK,
native/production qualification and panel results before any submission.

One serial numerical process at a time, nice10/BLAS1,8GB process-tree RSS,
256MiB maximum additional system swap; existing supervisor retires every
owned process group. Qualification includes exact parent and feature repeat.
