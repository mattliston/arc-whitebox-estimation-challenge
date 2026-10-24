# Qualify final-objective covariance-edge learning on current P

Phase2 only. Parent is the exact official331760 candidate, SHA256
8260ddb1f4966cc6acc4c318a53835981fde1b839d946ac10604b18325d2a040.
The completed dual-terminal information study remains failed; its labels,
folds and features are not reused to select this model.

The current bank trained its mean/GNN P parameters while keeping its existing
covariance-edge network fixed. That edge network was learned from local
moment discrepancies, not the current final native objective. This study
qualifies gradients for its five existing leaves: left[14,16], right[14,16],
edge[3,16], bias[16], output[16], totaling528 FP32 values. All P arrays,
normalizers, edge scales, birth-node A and cstar stay fixed. The forward
architecture and participant operations do not change at initialization.

This is related to prior covariance learning, not an untested general family.
The old native SPSA fit changed only16 output coefficients on six networks and
failed its fixed gate after36 paired updates. Earlier joint P/A and P/Q fits
also ran and retain their failures. This proposal instead uses the existing
approximate differentiable full-lower-moment translation to adapt all528 edge
values to the final objective on the established TRAIN160. It does not claim
to certify native derivatives or predict improvement.

Before any fit or new target array is opened:

1. Preserve an exact inverse from model_edge.py to the qualified model_v2.py.
   The only changes freeze P, enable the five edge leaves, and change the
   constructor's parameter checks. Bind the source to current331760.
2. Check a deterministic7-node covariance-edge fixture. Independent NumPy
   assembly must reproduce its forward result to relative2e-6. An analytic
   pullback through symmetric output, tanh, node and edge maps must match all
   five Torch gradient arrays to relative2e-5. No finite-difference search.
3. Run frozen35, then checkpointed edge-gradient35/100/35 under a synthetic
   final-output linear loss, with no mean-reference labels. Frozen/checkpoint
   and repeated35 outputs must be byte-identical. Both35 gradient dictionaries,
   all16 variance records,45 selectors and16 shared-state records must repeat
   exactly. Every leaf must have a finite gradient and the total norm nonzero.
   All parameter values remain byte-identical before and after backward.
4. Save the real last gradient's Adam boundary using the previously qualified
   generic state transport. Restore into new leaf objects; all parameter,
   Adam step/moments/settings and CPU RNG bytes must match. A second identical
   synthetic-gradient update must also match exactly. These state checks are
   unrelated to candidate selection and export no fitted participant.

Use one nice10 CPU group, all six BLAS limits and both Torch thread limits1,
7.5GB RSS emergency limit,0.05s polling,256MiB emergency swap stop, zero added
swap to qualify, and a2GB post-graph retirement limit. Copy the established
allocator and guard unchanged. Each full call is run serially in one guarded
qualification process with a1200-second ceiling; unit limit180s. Disk floor
470GB and new artifacts100MB. No GPU, sampling or download.

Only a complete qualification admits a separately sealed training protocol:
one fixed160-graph/40-update epoch, batch4, seed9409213, Adam cosine1e-3 to
1e-4, betas.9/.999, epsilon1e-8, zero weight decay, global norm cap5, loss
(final MSE +.05 earlier-layer MSE)/2e-8/4. Fresh process per optimizer batch;
exact state transfer; terminal payload only. This schedule is fixed here
before targets and is not selected using qualification synthetic gradients.
No intermediate checkpoint, epoch, sign, gain or learning-rate search.

Then actual native VALID40 must improve pooled own-cost adjusted error and
win at least24 networks, followed by the established independent14,
confirmation8, client, resource and package gates. Every existing failure
and official receipt is preserved. This qualification performs no fit,
production export, validation target access or upload.
