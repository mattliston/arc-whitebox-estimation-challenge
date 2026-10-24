# Frozen history covariance decoder: current native endpoint

Phase2 only, prospective after checkpoint86. Parent is official331684,
source facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0,
adjusted3.5844144212297072e-9.4e-9 banked;3/2/~1e-9 remain unmet.
The qualified covariance/readout composition is unchanged. Its previous
automatic upload rejection and existing unanswered approval question are
not bypassed by this local work. No upload is selected here.

## Distinct endpoint and preserved negative

The stored learned_pair_history_model_v1.npz has current SHA256
03fc240cf11f6b4246f6f644aa1447d93a962fc2ed343df8ce6be2623ff60776.
The original six-network/100-epoch fit on coherent8M oracle-input states
had held covariance ratios.7414121563/.7444433204 but next-mean-response
ratios.9933264149/.9855317053. It FAILED its joint<=.75 gate. Preserve that
result and the original decision not to write covariance. A bounded search
found the trainer and related terminal bilinear readout experiment, but no
native consumer of this exact stored model. This is not proof of global
absence, a new model family or a positive forecast.

This new study explicitly revises the old no-endpoint stopping decision to
measure actual final-layer loss times the complete native cost, just as the
separate frozen edge source was eventually measured. No extra epochs,
refitting, scale/layer/feature/rank/normalizer/precision search or combination
with a new readout. The related terminal2080-column additive readout's10%
regression and current nonlinear528 edge panel's0.96% regression remain
negative. Neither is this multilayer payload-transfer endpoint.

## Exact model and placement

Replace only the current covariance edge source at layers8..14. Keep the
same physical backbone, all other model parameters, layer policies and
final readout. Immediately before that covariance-source addition, form a
separate sidecar state from the current uncorrected physical m,C,K21,aux,W.
Use the current features() schema and gnn_step() equations at layers0..14,
with the ORIGINAL frozen P0 payload_compact_feed_fullpair_v1.npz,
SHA166b3c67ec4d27793d433e38e15086b95d052426fdf435a79067cc07a2a4dd7e.
Initialize H0 from a fresh zero state per prediction. Its mean delta is
discarded. Its hidden state NEVER substitutes for the parent's mean GNN.
The current source's final-layer mask is irrelevant because this sidecar
does not run at15. Qualify its equations against the original NumPy source.

For current H(n,64), sigma and pre-correlation R, append a constant to H:

  F = ([H,1]/feature_scale) * (sigma/sigma_scale)[:,None]
  E = cov_scale * sum_(r=0..2) (F @ theta[r] @ F.T) * R**r.

Use the saved symmetric theta(3,65,65) and normalization fields without
changes. The original scalar sigma_scale/cov_scale are rounded toFP32,
matching the original Torch FP32 arithmetic. H,F,R,theta and the three
products/assembly are FP32; cast E toFP64 for the existing covariance add.
Keep its diagonal. No new symmetrization, clipping, PSD repair or floor.
Floating-point symmetry is qualified to3e-5 relative, not assumed bit-exact.
The old edge is REPLACED, not added again. Current autonomous input states
differ from the original oracle-state training law; that shift is measured.

All deployment arithmetic is legal FNP within prediction. Static bytes are
decoded by the parent's existing lossless loader inside prediction. The
additional recurrent state is local to each call, not a cross-network cache.
Disabled mode must inverse to the exact parent and preserve its full bill.

## Provenance and target-free qualification

The model file was not cryptographically pinned by its old verdict. Freeze
its present bytes before decoding; do not call them a historical sealed
training endpoint. Corroborate them by replaying BOTH recorded old VALID
400/500 covariance and next-response losses using saved histories and source
arrays, requiring relative agreement<=1e-4 with the final old verdict.
This reads existing8M labels only to check the fixed archived model; there
is no fitting, selection or new raw-error estimate. It cannot prove unique
parameter identity from loss equality. Authenticate the old P0, generator,
trainer and history receipts. Do not access ordinary1B development labels
until the new native predictions and complete bills are sealed.

Before full calls, verify source inverses, every exported tensor byte and
positive scalar/feature normalizers. Test actual FNP kernel versus independent
NumPy FP32 and FP64 formulas on three deterministic small fixtures, including
symmetry, node permutation, repeated calls, cold/warm parameter setup and a
zero theta. Relative tolerance3e-5; zero theta exactly zero. Test actual
current features and history gnn_step against the original NumPy feature/GNN
functions at0/1/8/14. Hidden state relative<=3e-5; feature scaled error<=3e-6.

Exactly seven full native qualification/development calls: disabled35/100/35,
enabled36/38/36, and separate cold38. All original disabled output bytes,
full bills and weight hashes must match their saved exact-parent controls.
Enabled first8 output rows must equal parent. Own36 repeat and warm/cold38
outputs and captured state/source arrays must be byte-exact. Retain passive
references only; no extra FNP operations in the measured call. Check all16
unfloored variances>=1e-10 and all outputs/sources finite. Existing native
numeric caches must retire. Save small per-layer features/histories and the
fixed16-node source submatrix0,67,...,1005; no wide covariance tape.

Replay sidecar recurrence from its actual saved inputs using original NumPy
P0 equations, relative hidden-state<=3e-5 at every layer. Check all saved
source submatrices against independent FP32/FP64 formula<=3e-5 relative.
These fixed-input checks do not certify a derivative of the physical loop.

## Actual objective and advancement

Networks36/38 are outside the old six TRAIN and two VALID nets but reused in
broader research. They are not pristine holdouts. Compare against exact331684
saved cold outputs, each arm with its own complete measured cold bill and
max(.1,F/2**41) multiplier. Seal qualified predictions before ordinary1B
mc_post references. Require pooled adjusted ratio<1 and each<=1.05.
Independently check MSE with math.fsum to relative2e-12.

If negative, close this fixed endpoint without tuning, another fit, panel,
package or upload. If positive, a separately implemented fixed14 panel
[40,41,43,44,45,46,47,49,50,51,52,53,54,55] must have pooled<1,>=8wins,
worst<=1.10; confirmation[37,39,42,48,59,63,66,67] must have pooled<1,
>=5wins,worst<=1.10. Actual client/runtime,contract,package and exact external
authorization remain necessary. A local pass does not complete the goal.

## Resources

One serial nice10 CPU group, six BLAS variables1,8GB group RSS ceiling,
256MiB emergency swap cutoff and ZERO added swap required. No GPU, new
samples, dataset copies or downloads. Build60s, old-model replay180s,
units120s, each native sequence300s, qualification180s, score60s. New
artifacts<=250MB. Preserve all errors and prior sources. Stage sources are
sealed before execution; later code completion cannot change this design.
