# Final variance-source information, unchanged propagated estimator

Phase2, September5,2026,23:04PDT. One fixed final-head information pilot.

Current final Q sees total pre variance and scalar skew/kurtosis, but not
the covariance contributions aligned with the ACTUAL final weight vector.
Earlier48 plane features were row/column contractions with eight NODE
functions; the learned graph mixed node embeddings. Neither computed this
next-weight quadratic read of separated covariance response components.
The older mechanistic31 feature set zeroed the final layer and likewise
did not expose these current on-policy final-weight quadratic forms.

At post14 only, split the EXISTING full-pair covariance into Gaussian
correlation orders1..6, third-cumulant response orders0..2, fourth-cumulant
response orders0..2, exact post diagonal, and the current predicted-mean
feedback update. There are14 components. They sum to the current actual
post covariance after its unchanged active mask. Push each component
quadratically along W15 and divide by the current final pre variance.
No new physics, targets, state writes, support changes or moment samples.
Feature value is tested, not assumed from the identity.

Target-free prerequisites: synthetic dense pair identity<=1e-12; current
rank35/100 component covariance sum relative<=1e-12; final variance sum
relative<=2e-6 with the existing float32 covariance boundary; full original
NumPy trajectory/h/host/sigma/head output unchanged byte-for-byte by the
instrumentation. No tolerance loosening. No labels until these pass.

Then exactly24 rows35..58 from the frozen dynamic P prefix and scalar-final
Q. Every row must pass the same parity and decomposition checks. Train20,
holdouts37/39/42/48. Compare a final Q extended126->140 inputs against the
identical extended Q with all14 inputs zero, same initial old Q weights,
zero weights on new columns. All final-head parameters may co-adapt;
prefix P is frozen and no recursive retraining is involved.

Fixed seed930062,40 epochs, AdamW5e-4 cosine to5e-5, weight decay2e-5,
batch8192, clip gradient norm5, final epoch only. New-feature center/scale
from TRAIN ONLY, min scale1e-6, clip+-8. Labels are existing1e9 final means,
opened only after the input-generation gate. No layer/order/feature subset,
epoch/learning-rate/radius/seed sweep. Require pooled final raw MSE<=.90
of the current bank AND<=.92 of the zero-input control, and3/4 bank wins.
Failure closes this source-information pilot; no200 rows or deployment.

Production upper envelope before optimization: twelve dense float32
1024x1024 component matmuls plus diagonal and rank-two feedback quadratic
reads only at the final boundary; source elementwise maps and140-input
head. Dense work~26B classical FLOPs, <6% of the current490B bank, plus
source elementwise construction. This is NOT a measured deployment bill.
A pass still needs full200 fixed-split fit, complete actual cost,14-net
independent panels, persistent-runtime/memory and package checks.

Use at most two nice10 CPU row workers, two BLAS threads each; one MPS
fit at a time only after rows finish. AP Torch never imports threadpoolctl.
Preserve every output, no commits, no forum writes, no unqualified upload.
