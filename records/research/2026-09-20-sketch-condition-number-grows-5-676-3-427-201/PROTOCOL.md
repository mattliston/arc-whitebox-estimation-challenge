# Current shared sketch numerical diagnostic

The official parent is submission331684, SHA256
facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0.
Previous goal turn68 made progress by completing a negative fixed-selector-order
intervention. That failed estimator and its response/refit gates stay closed.
This study changes no estimator. It measures the ORIGINAL parent's internal
range sketch and the already executed head0 perturbations on rank38.

Deeper source review found BACKEND_DECISION_20260912.md: NumPy QR already
widens FP32 inputs internally. The installed _linalg.py confirms this. Merely
declaring QR64 would increase its bill without improving its factorization.
Neither the previous conditioning caveat nor that review contains a measured
current-parent range-sketch conditioning result. This is a diagnostic of the
FP32 sketch producer, not a precision rescue of the canonical-order candidate.

Exactly five cold full calls: original rank38 base, head0 plus1/16, minus1/16,
plus2/16, and base repeat. Each must reproduce ALL16 saved output rows and
the complete original bill after subtracting explicitly measured snapshot
copies. No labels, fitting, parameter changes, candidate, package or upload.

Wrap the existing native QR call only. Before calling it, copy Y and its
current producer operands J, weight, Omega, previous row indices, Qp and core
(last two only when present). After QR copy Q and R. Every copy uses FNP and
is charged inside the same counted prediction. The original QR receives the
original Y. No array is converted to NumPy during prediction. Unit fixtures
must prove independent snapshot storage after input mutation and exact QR
outputs, including strided/ill-conditioned inputs. The wrapper checks its
caller is the final _shared_old384_push and records exactly layers5..14.
All conversion/reference arithmetic happens after the counted call. Hold
only these small matrices, never full CP/physical covariance trajectories.
Replayed output/bill and repeated trace bytes are mandatory qualification.

After capture retirement, analyze one layer/run at a time in FP64. Compute
QR reconstruction and orthogonality, all singular values, and the projector
from native Q. Independently reconstruct the SAME frozen-input sketch as
J diag(weight) J.T Omega + Qp core Qp.T Omega, with FP64 GEMMs. Captured
weight is already rounded FP32: this reference isolates subsequent matrix
arithmetic, not coefficient-square rounding or error in earlier states.
Compare its QR projector and input sketch to the native ones. Also refactor
the captured native Y in FP64 and require rounded Q/R byte equality; this
checks the installed QR widening fact on the actual calls.

Scatter bases/sketches into their physical1024 rows before cross-call
comparison. Projectors, not individual Q columns, remove QR sign/basis
ambiguity. Report centered/forward and double-step response discrepancies
per layer for native sketches, native projectors, same-input FP64 projectors
and saved output means. No finite-difference gradient theorem is claimed.
An ill-conditioned or rounding-sensitive sketch does not establish a causal
end-to-end accuracy benefit. Any implementation change requires a separate
declared objective experiment with actual own cost and held network checks.

Small dense fixtures compare the factored FP64 reference with an explicit
Gram construction, QR residuals and row-permutation-invariant projectors.
Preserve any implementation failure and version a repair; never overwrite
receipts or relabel a failed qualification as a pass.

One nice10 numerical process tree, six BLAS limits1, 8GB RSS, require zero
added swap, guard120s per full call and300s analysis. No new datasets,
downloads, GPU, model fitting or truth decoding. Total artifact cap1GiB.
Existing parent, perturbation sources and all prior receipts remain intact.
