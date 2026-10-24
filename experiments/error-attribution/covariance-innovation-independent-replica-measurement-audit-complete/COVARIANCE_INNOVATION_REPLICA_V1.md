# Independent-stream covariance-innovation reproducibility diagnostic

Phase2, September5,2026. This is a measurement-quality audit, not reopening
any closed predictor or claiming an estimator gain. No fits or candidates.

The eight source-only covariance rows use coherent8M previous empirical
means and covariances. Their fresh L8..14 covariance innovation on35 has
entry MSE~3e-9. Its two existing cov/cov2 files are identical, not replicas.
The current-parent accumulated covariance error's old noise estimate does
not establish the noise level of this different oracle-state innovation.

Generate ONE fresh independent8M input stream for rank35, using the
read-only existing `mc_truth_mps_indep.py` numerical implementation with
PCG64DXSM seed `(mlp_seed xor 0x4153545241434F56) mod 2^64`. This differs
from the original0x47414D454F4E5F31 stream and previously named streams.
Keep its131072 batch size, MPS float32 matmuls and per-batch CPU float64
accumulation. No threadpoolctl in the Torch process. One nice10 MPS job,
no concurrent card/training jobs; expected~3minutes, based on prior logs.
Fresh output/provenance paths only. Never overwrite the original truth.

Replay BOTH original and replica oracle-previous-state chains, with the
same parent and feature/target timing as the source-v2 protocol. Original
E must reproduce frozen v2 targets byte-exactly. Save fresh innovations
BEFORE current mean/covariance feedback. A second diagnostic channel
records raw-second-moment error (equivalently, covariance error AFTER the
oracle's consistent mean correction); do not interchange the channels.

For L8..14 and pooled, report

  observed energy = (||E_A||^2+||E_B||^2)/2,
  shared signal point estimate = <E_A,E_B>,
  per-stream noise point estimate = ||E_A-E_B||^2/2.

Use the same fixed target-free current-parent next-layer response and W
for BOTH streams' next-mean projections. Also separate diagonal/offdiagonal
and the covariance's one-high mean-direction sector, using the CURRENT
PARENT mean, not an oracle-selected direction. No model coefficients.

Two streams give point estimates, NOT a confidence-certified ceiling.
Nonlinear empirical-state feedback can introduce O(1/N) bias. MPS and
float32 covariance-file arithmetic are shared and are not measured by
stream differences. Qualify all such limitations in the verdict.

If rank35's pooled noise fraction is >=20% in either the covariance or
next-mean BEFORE-feedback channel, run the same one-replica protocol on
rank100 for a cross-net check. Otherwise stop after35. No sample-size or
seed sweep, no automatic model retraining or gate relaxation follows.
