# Fixed full-matrix covariance / third-moment information test

Phase 2, 2026-09-19. Parent is the approved Lazy source
b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c,
submitted as #331599. Its source, archive and physical predictions stay unchanged.

Question: do full two-step operators of the current post14 covariance C and
repeated-third plane S contain useful remaining terminal-mean information?
The archived TWO_HOP_JOINT_OPERATOR_SOURCE_AUDIT_V1.md did not locate a value
test of full RS/SR operators. Existing one-hop learned graph, covariance-energy,
pair-feature, terminal-representation and covariance-control negatives retain
their actual scopes. This new test is not a new cumulant closure or a claim
that these products equal a missing physical source.

Observe C as received by the final covariance push and S as the actual post14
target after its original active mask and diagonal restoration. The latter is
the parent's repeated-plane ownership convention; do not call it a newly
recentered true third moment. Also observe existing final hidden64, sigma,
alpha, pdf and final W. Add no arithmetic inside the parent prediction.

Let s=sqrt(max(diag(C),1e-10)), R=sym(C/(s outer s)) with zero diagonal,
T=S/(s^2 outer s) with zero diagonal, and U=diag(s) W. These are diagnostic
feature definitions only; symmetry and the floor do not modify the parent.
Cast R,T,U to FP32 and compute A=R U, B=T U, D=T.T U with three full paid
matrix products. Define five output vectors, dividing each by final sigma^2:

    sum(A*B), sum(A*D), sum(B*B), sum(D*D), sum(B*D).

They are diagonals of U.T R T U, U.T R T.T U, U.T T.T T U,
U.T T T.T U and U.T T T U, respectively (R is symmetric; quadratic forms
implicitly select symmetric parts). Reductions widen stored products to FP64.
Ten fixed features are these five vectors times pdf, then times alpha*pdf.
No feature, derivative, layer, strength or operator selection after targets.

First certify the formulas independently against explicitly formed small
operator matrices, including nonsymmetric T, a diagonal-only zero case and
finite FP32 association error. Then observe full35/100/35: require exact
all16 parent outputs AND original parent FLOP bills on every call, plus exact
repeat features/hidden/sigma/predictions. Meter features in a separate context
and require less than9GF per call. Each resulting screen score charges parent
plus its measured feature bill and a2MF readout reserve. That sum is an honest
information-test cost, not a qualified production candidate bill.

After qualification, capture the fixed TRAIN20 ranks
35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58,
each in a fresh process with a complete cold parent bill; reuse qualified35.
Only compact features and outputs are stored, not full covariance tapes.
Seal every capture before opening any ordinary1B mean reference. These are
reused research networks, so the result is exploratory network cross-validation.

Use the already certified terminal_representation_v1 weighted ridge method:
five folds by network ordinal modulo5, TRAIN-only RMS normalizers, ridge1e-4,
control sigma*[1,H15], and TRAIN-only weighted SVD projection of the ten
extensions off that65-column span (relative cutoff1e-7, residual RMS cutoff1e-6).
This removes duplicate-column regularization effects. Preserve all fitted
fold objects and seal FP32 nonnegative held-out outputs before aggregate scores.
No optimizer, backbone gradients, checkpoint or penalty search.

Advance only if pooled adjusted <=.95 of unchanged and <=.97 of affine
control, with at least4/5 fold wins against both. A failure closes this fixed
feature family; no CHECK targets, full fit, extra features or native deployment.
A pass needs a separate TRAIN fit, untouched-by-this-study CHECK and full
native/client/panel qualification before a bank. Approximately1e-9 remains the
actual goal; this information test is not a replacement completion criterion.

One nice10, BLAS1 numerical process group at a time, existing8GB RAM and
256MiB added-swap watchdog; successful qualification requires zero added swap.
Qualification whole-wall180s, each later capture100s, scorer120s. New data
allowance20MB. No GPU, no large downloads, no external messages or new upload.
