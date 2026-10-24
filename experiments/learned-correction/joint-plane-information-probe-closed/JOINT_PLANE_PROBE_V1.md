# Bounded joint-plane information probe

Phase 2 only, September 4, 2026 PDT. No bank changes or new upload.

The closed graph round used covariance-neighbor alpha/gate/pdf summaries,
not response-weighted contractions of the current third-cumulant plane.
The closed tensor-direction round used nine full-tensor directional reads
on the earlier compact host. This probe asks whether the current full-pair
host's *joint alignment* of K21 with covariance and ReLU response contains
information absent from its four existing K21 row/column summaries.

Fixed 48 features: six off-diagonal standardized kernels (T21, its transpose,
each multiplied elementwise by rho, rho squared, T21 times its transpose)
applied to eight node functions (1, gate, pdf, alpha*pdf, (alpha^2-1)*pdf,
skew, kurtosis, host/sigma). No current-correction-updated state or labels
enter these features. All parent hosts, sigma, and 30 old features must be
byte-identical before saving any row.

One cheap information gate before any 200-net retraining: ranks35..58,
same fixed held-out identities37,39,42,48; frozen parent mean-head residuals
on honest teacher-forced rows. Compare a fixed regularized linear residual
probe using old features against the same probe augmented by the 48 new
features, with identical fixed local response interactions. Pool layers10..15.
Require at least5% extra held-out residual-MSE reduction before considering
a full matched GNN round. This is an information diagnostic, not a learned
closed-loop accuracy or release claim. No hyperparameter sweep.
