# Lossless static FP32 packing, fixed implementation

Selected after profile_qualified.json: unchanged predictions/bills and a
.131024625s cold module-load component. Replace only the representations of
_GNN_LITERAL and the older _FINAL_LITERAL, plus their three corresponding
FP32 initializer comprehensions. Flatten each immutable literal in C order,
pack its exact IEEE little-endian FP32 bytes, zlib-compress and base64-encode
at build time. In the candidate use standard-library base64/zlib decoding,
public fnp.frombuffer(float32), reshape to the original shape and copy to an
owned array. Decoding and array construction happen in the ordinary predict
budget. Preserve inactive legacy functions rather than changing dispatch.

An inverse edit must restore all parent source bytes. Validate every decoded
parameter byte against the old FP32 conversion and verify actual FNP arrays.
No physical arithmetic, learned coefficients, thresholds, amplitudes, layers,
GC policy, accounting, timing exclusion or deferred work changes. No NumPy
import in the candidate; no sidecar weight file or downloaded data.

Native35/100/35 plus cold100 must have all16 output bytes exact to331676,
exact repeats and warm/cold100 output equality. Complete warm bills must be
unchanged; any cold array-construction bill change is recorded and must be
within0.1% of parent. No accuracy target is needed to assess a byte-identical
representation change.

Then measure contemporaneous unprofiled real-client parent/candidate cold
pairs on fixed ranks[35,100,37,39,42], using a separate process for each cold
call. Alternate pair order by roster index. Require all outputs exact, actual
bills consistent with native qualification, each candidate residual<=.25s,
and mean candidate cold residual<=.75 of matched parent's mean. Also run
candidate35/100/35 in one process: exact outputs/repeats, residual<=.25s for
every call, and no material wall regression (candidate/parent pooled wall
<=1.075, each<=1.10 on matched cold pairs). These are local gates, not a claim
that platform module loading is identical or that official failures are fixed.

If negative, preserve and diagnose; do not blindly upload. If positive, qualify
contract/package and consider an authorized official trial. Runtime repair
does not promote the failed official331676 score into a successful bank.
The covariance candidate is a separate model and requires its own runtime
qualification before any combined or standalone upload.

One guarded numerical group at a time,8GB RSS,one BLAS thread,zero added swap.
No new training, targets, samples, GPU or dataset downloads.
