# Fixed terminal nonlinear bagging information screen

The preceding turn made progress: Weight5 qualified and the fixed Read6
development experiment closed negative. Official best remains#331584 at
4.015535762022727e-9; Weight5 is only locally qualified, with exact archive
approval pending. This experiment does not change or upload those sources.

Question: can an average of small nonlinear fits extract a stable terminal
residual from the existing encoder input, while reducing fit variance? The
earlier H15 quadratic and382-channel linear-span extensions were negative;
the original full-carrier44,144-parameter single gradient-direction trial
was also actually measured. This is a separate4,097-parameter terminal-only
residual network, with frozen captures/physics and no recurrent derivative.
No claim that prior negatives prove a universal representation limit, or
that bagging guarantees a useful correction.

Reuse the exact20 original Read5 captures under
`../terminal_representation_v1/train_capture/manifest.json`, whose source,
weights, targets and qualification were sealed earlier. Capture arrays are
unchanged. No additional full estimator call, sampling or large dataset is
needed. TRAIN ranks in increasing order are35,36,38,40,41,43,44,45,46,47,49,
50,51,52,53,54,55,56,57,58. Five folds are ordinal modulo5. These networks
and targets have prior model/research history: this is exploratory network
cross-validation, not independent generalization evidence.

Baseline is the saved served prediction. Affine control is the previous
fixed sigma*[1,H15] ridge1e-4, with TRAIN-only RMS normalization and each
parent's own cold multiplier as sample weight. Recompute it and check its
predictions against the previously sealed fold controls. SVD cutoff1e-7
defines the TRAIN control-span projector. An independent algebra/gradient
certificate must pass before the target loader is allowed to run.

New input is the actual126-column terminal encoder input. Per outer fold,
center and standardize using weighted TRAIN means and standard deviations,
floor scale at1e-6, then clip to[-8,8]. No held-out statistic is used.
Set r=max(2^-20, weighted TRAIN RMS((truth-affine_prefloor)/max(sigma,1e-12)))
and round r once toFP32.
One member f has126->32->1 layers, SiLU, exactly4,097 parameters. Its raw
correction is q=sigma*r*f(input). Initialize first-layer weights independently
uniform[-1/sqrt126,+1/sqrt126], first-layer bias0, output weights/bias0.

At EVERY training step project q off the common TRAIN control span:
beta_q=pinv(sqrt(w)*Xn)@(sqrt(w)*q), delta=q-Xn@beta_q. This is part of the
differentiable loss. Evaluate held-out delta using this SAME TRAIN beta_q.
It prevents the branch from merely refitting the affine control. The small
projection can be folded into the final readout for a later deployment;
it is not a large matrix required at inference. TRAIN projection coefficients
are saved per member; no held-out labels enter them.

Four members per fold. Initialization uses independent PCG64DXSM seeds
190919700+16*fold+member. Network bootstrap uses190919900+16*fold+member:
draw exactly16 indices with replacement from that fold's16 TRAIN networks.
Its counts multiply the actual parent-cost sample weights. Normalizers and
the affine/projector fit use all16 outer TRAIN networks, shared by members;
bootstrap changes the member's supervised loss weights. This is not a claim
that omitted bootstrap networks were wholly unseen by the affine control.
Seal all20 initializations and network draws before opening any target.

Each member trains exactly200 full-batch AdamW updates, FP32 model tensors,
FP64 control projection and squared-error reduction. LR follows cosine from
1e-3 to1e-4 over updates0..199, betas(.9,.999), eps1e-8, weight_decay2e-4,
foreach=False, global gradient norm cap5. Loss is bootstrap/cost-weighted
MSE of clamp(affine_prefloor+delta,0) cast toFP32, against original stored
FP32 truth widened toFP64, divided by the fixed initial affine weighted MSE.
Autograd treats floating casts in the usual straight-through sense; this is
not a certified derivative of native rounding or the physical backbone.
Require finite loss, gradients, parameters and optimizer state. No validation
early stopping, selectable checkpoint, seed/ridge/width/LR/epoch sweep or
extra full-data fit follows a failure.

Save and reload each terminal payload, then form its held-out correction.
The ensemble averages four corrections BEFORE the final nonnegative FP32
serve. Individual-member predictions are diagnostic, never selectable.
Seal every fold's held-out predictions and all20 terminal payloads before
aggregate scoring. Compare unchanged, affine control and ensemble. Include
1MF readout reserve for affine,20MF per single member,80MF for the four-member
ensemble above each actual parent cold bill. These conservative reserves are
not a measured deployment bill. A future native port must pay the whole model.

Gate: ensemble pooled adjusted <=.95 of unchanged, <=.97 of affine, with
at least4/5 fold wins against both. This is an information gate, not the
approximately1e-9 goal or a bank. If it fails, close this exact nonlinear
bagging hypothesis without CHECK access, more epochs, architecture/penalty
search or a full fit. A positive only permits separate CHECK/native work.

Report mean individual loss and ensemble spread as diagnostics; verify the
MSE(mean)=mean(MSE)-mean(variance) identity on unclipped FP64 predictions.
Clamping/FP32 serving need not obey that identity exactly. Also report training
orthogonality and train/held-out losses to expose overfit. All outer-fold
predictions use TRAIN-only normalizers, parameters and projection coefficients.

Only one nice10 CPU worker, all BLAS/Torch threads1, existing8GB RSS and
256MiB additional-swap guard,900s wall cap. No MPS/GPU, full network tensors,
backbone recomputation or added Monte Carlo labels. Preserve scripts, seeds,
payloads, traces and every failed gate. No upload is part of this experiment.
