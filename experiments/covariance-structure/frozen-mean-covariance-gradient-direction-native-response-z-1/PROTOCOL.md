# One covariance-only direction, calibrated by native forwards

Selected 2026-09-20 after the exact Lazy cost census and review of the
completed joint P/Q fit, its backend-gap diagnosis, the 12-direction coherent
scale test, and the terminal covariance65 and bilinear2080 negatives.
This is a new bounded optimization experiment, not a learning-rate or epoch
extension of the old joint fit. Those endpoints and gates remain unchanged.

The question is whether ONE pooled covariance gradient proposal, with every
existing learned parameter frozen, provides a useful native response on the
accepted Lazy parent. The old COMP MPS model supplies a proposal only. It is
not the Lazy native circuit or a qualified native gradient. Source admission
found exactly the final64 head coefficients and one final bias differ from
COMP. The proposal uses ALL actual Lazy P literals, frozen, with unchanged
normalizers and layer0 inactive coordinates. NODE A and cstar are identical.
This correction was made before source sealing, numerical imports or runs;
the original source-admission premise is preserved separately.

The action is the previously implemented C-only diagonal congruence after
post-layer mean recentering, active mask and diagonal restoration, before
Bpost/trace/innovation/star construction, at layers1..14:

    z = H @ Q[layer-1,:64] + Q[layer-1,64]
    s = exp(.005*tanh(z)) in FP32, widened to FP64
    C = (C*s[:,None])*s[None,:]

All future state and selectors remain live. This is a closure compensator,
not a coherent all-cumulant scaling or PSD repair. No intervention at0/15.
The whole Lazy source must inverse exactly after removing these edits and
the helper/payload. No mean, birth, rank, kernel, gate or normalizer change.

FIT is exactly35,36,38,40. These are reused public TRAIN references, not fresh
generalization. CHECK, if reached, is37,39,42,48. No CHECK truth is opened
during proposal, response qualification or FIT calibration.

Proposal: exactly one MPS forward/backward per FIT network at Q=0, P/A/cstar
frozen. Loss is the mean over four networks of final served MSE/2e-8, weighted
by each saved current-Lazy cold multiplier divided by their mean. No earlier
layer loss, optimizer step or checkpoint selection. Accumulate gradients in
FP64 on the host. Save each gradient, baseline output and each original H.
Direction is minus the pooled gradient, divided by the largest absolute
affine H response across all four baseline trajectories and14 layers.
Require a finite nonzero gradient and normalization. Seal direction before
native trials. This normalization sets baseline |H d + d_bias|<=1, not a
guarantee about changed or unseen trajectories.

The sole radius h=1/32 is fixed now. Evaluate native Q=t*d at t=0 on35 and
t=+h,-h,+h/2,-h/2 on each FIT network, serially. Null35 must match ALL16
accepted-parent bytes. Repeat +h/2 on35 must be byte-exact. First two output
rows of every nonzero call must equal the parent. Original complete bills,
cache retirement, finite/nonnegative outputs, and all14 action visits must
be recorded. Require |z|<=1/16 at every actual write; otherwise stop. The
direction is not rescaled after native results.

For each network let r_h=(y(+h)-y(-h))/(2h), r_half analogously. Require
||r_h-r_half||/||r_half||<=.05, RMS(h*r_half)>=5e-6, and
||y(+h)+y(-h)-2*y0||/||y(+h)-y(-h)||<=.10. These are two-amplitude response
stability tests, not a proof of all native derivatives or selector smoothness.
Stop on failure; no radius/direction/precision/roster rescue.

If qualified, use r_half and the literal saved native parent outputs and
ordinary1B final references. Compute one weighted scalar ridge solution
t0=-sum(w*r*e)/(1.001*sum(w*r*r)), clipped to[-h,h]. No search. The predicted
weighted final SSE must be<=.80 of parent, and every network<=1.00. A failure
closes this direction only, not covariance capacity or all solvers.

Only a pass permits one actual native FIT evaluation at that t. Require
own-complete-cold-bill adjusted ratio<=.85 pooled and <=1 on every FIT net.
Then unchanged Q is tested on CHECK4, requiring adjusted<=.90 pooled and
each<=1. Only a pass warrants the fixed existing NC14 panel, confirmation8,
real client <=45s and pooled wall<=1.10 accepted Lazy, contract/package checks
and an authorized board submission. Downstream stages need concrete sources
and seals before execution; no adjustment of this proposal follows failure.

Resources: one numerical group, nice10, six BLAS thread variables1; guarded
RSS<8GB and zero additional swap for qualification. MPS allocator hard cap
4.5GB; proposal guard180s. Native calls each90s, no retained model across
processes; one repeat pair may use180s. Projected files<30MB excluding sealed
small candidate copies (<12MB). No new samples or downloads. Preserve every
failed result. All saved truths/predictions remain research-only.

## Version2 runtime recovery before any completed proposal

Version1's unit algebra/source checks passed, but the first incomplete MPS
forward ended in a libdispatch allocator trap. Preserve its full closeout.
Version2 keeps the SAME4.5GB device allocator and8GB RSS caps. It synchronizes
and clears cached device buffers at each completed forward/replay layer,
records driver allocation there, and enables Python faulthandler. No model
source/arithmetic, parameters, radius, labels, loss or selection gates change.
The original exact native helper unit certificate is reused by source hash.
The entire four-network proposal is computed once to completion if possible;
no parameter/direction survived the interrupted first attempt.

## Version3 final bounded runtime recovery

Version2 completed all16 forward steps on35 with3,257,647,104 driver bytes
at the last logged boundary, then terminated in the MPS allocator without
a completed backward or saved direction. The actual peak need is not known.
The second matching crash report and all artifacts are preserved.

Version3 changes ONLY the device allocator/driver limit from4.5GB to8GB,
keeps per-layer cache cleanup, and adds a before-backward progress receipt.
The independent host RSS cap remains8GB, with the same swap guard and zero
additional swap required. The48GiB machine can accommodate this bounded
device allowance without an unlimited allocator. No installed-library edits.
No further MPS limit/cache variants are selected if this attempt fails.
The same fixed four-network proposal, labels, actual Lazy frozen parameters,
native direction/radius/response/value gates and no-optimizer policy apply.
This supersedes only the4.5GB resource sentence above.

## Version4 source-seal repair

Version3 never imported numerical libraries: its source inventory mistakenly
included its own just-created empty seal. Version4 constructs the inventory
before creating the seal and explicitly excludes that filename. Every
executable byte is identical to Version3. This is the same final8GB-device
attempt, not a further allocator/parameter/scientific variant. Preserve the
Version3 admission failure.
