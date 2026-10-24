# Fixed covariance model with lossless static packing

This is a representation-only derivative of the accuracy-qualified frozen
covariance source04dd7586ac6826f9aef65d873b719d05184355dd2dc6552f32d6046c066d0180.
It is conditional on readout_residual_runtime_v1 passing its predeclared
packing and ordinary-client runtime gates. Do not start numerical work before
that outcome and retirement of its guarded groups.

Reuse the exact GNN/legacy-FINAL FP32 packing implementation: byte-exact
little-endian values, base64/zlib, public FNP frombuffer/reshape/copy. The
covariance tensors, readout, correction amplitude/layers and all arithmetic
stay unchanged. Inverse source restoration and every parameter byte must
pass. Keep this distinct from the packed readout candidate; do not combine
its fitted readout coefficients with the covariance source.

Native36/38/36 and separate cold38 must match the existing covariance source
on all16 output bytes and complete cold/warm bills. Keep exact own repeats
and warm/cold equality. No target re-scoring or fitting is needed for a
representation-only derivative that meets these checks.

Ordinary NumPy-free real-client cold pairs use[36,38,37,39,42], alternating
pair order. Compare runtime to the unchanged official331672 full-lower source;
require pooled total-wall ratio<=1.075 and each<=1.10. Require candidate residual
<=.25s in every cold and warm call, and candidate mean cold residual<=.75 of
the paired official source. Candidate outputs and complete bills must match
saved uncompressed covariance native outputs; first8 rows must equal the
paired official source. Candidate36/38/36 in one client must also reproduce
the saved native sequence exactly, including bills. These local gates do not
prove official resource failures are fixed; an official trial is still needed.

If any gate fails, preserve it and do not upload this source. If all pass,
contract/package validation then submit under standing authorization. Carry
forward the original fixed model's development,14-network panel and8-network
confirmation results as the unchanged model's evidence; those networks are
reused public research networks, not pristine holdouts. No new training,
accuracy targets, samples, GPU, downloads, timing/accounting changes or work
deferred outside prediction. Serial nice10 guards,8GB RSS,one BLAS thread,
zero added swap. Preserve the existing official bank until a better official
grade is recorded.
