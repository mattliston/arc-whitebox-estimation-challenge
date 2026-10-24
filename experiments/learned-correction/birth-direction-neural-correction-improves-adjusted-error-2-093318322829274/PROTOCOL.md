# Node-conditioned newborn orientation: prospective bounded pilot

Phase2 only,2026-09-08 UTC. This is a new state-action extension of the
successful frozen birth_direction_fit_v1 result, not an extension/refit of
that completed fit. Preserve all previous sources, receipts and closeouts.
No forum posts, commits, promotions or milestone-policy changes.

## Two routes, evidence and decision

The current4096-pair autonomous P-head fit is the selected immediate route
to a cheaper competitive card and is already running under its own frozen
protocol. Actual complete two-net mean cost311.663B F is33.52% below its
matched bank; unadapted raw5.54e-8/6.28e-8 is adverse evidence. It must
recover roughly2.12e-8 raw at that multiplier for3e-9. Its preset terminal
result, not an optimizer/epoch/rank sweep, decides whether it advances.

The strongest separate raw route is neuron-conditioned orientation of the
four now-successful newborn directions. Frozen layer coefficients reduced
native40 raw by5.19%, and the independent14 actual NC-adjusted objective
by5.23% on public8 and7.37% on virgin6. Changing how physical directions
are created has therefore earned a small cross-net gain. Earlier scalar
pair-amplitude conditioning and mean-head refinements did NOT test this
newborn orientation action; their negative receipts remain valid.

There is no measured evidence yet for additional neuron-level coefficient
variation, nor for the existing P state identifying it. No defensible point
forecast is available. Plan for a component-scale gain, not a milestone:
at current complete cost, the4.484e-9 independent14 reference needs another
10.79% adjusted reduction to cross4e-9 and33.09% to cross3e-9. This small
head is not currently evidence for either crossing, much less1e-9.

Cost is favorable: all current birth-direction work remains payable,
roughly467.683B F on the independent14 mean. At unchanged trajectory
geometry, fourteen1024x64 by64x4 readouts plus64-column normalization,
clipping, coefficient construction and256-F parameter initialization
fit a conservative0.025B-F extra allowance. No new CP columns, carrier,
n-by-n features or large matrix product are added. Mask changes can alter
the other work, so this is a planning inventory, NEVER a measured bill.

Decision: admit SOURCE implementation and fixed numerical fidelity checks
now, followed by the small fixed pilot below only after fidelity passes
and the sole MPS slot is free. This is the cheapest direct objective test
of the missing conditional-information assumption. Do not jump directly
to another160-network fit on positive scalar-coefficient evidence alone.

## Fixed causal law and compatibility

Parent is birth_direction_fit_v1/build_terminal_v1/estimator.py,
SHA d50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451.
Its independent14 result SHA is
767cf92415a0391f7b7596c9681adede144bd10b3e5d36f354ec2dce7ce52c1b.
Freeze P, the full carrier, and the EXACT stored float32 coefficients from
training_v1/terminal_theta.json SHA
de0eb2c2b3dbda9cef1e6d31146755f18f400aaa47f793460017fd9040753337.
Do not recompute the coefficient tanh in Torch and silently change bytes.

Let H_l be the existing1024x64 P state returned by the current layer's
gnn_step, computed from the PRE-current-correction features. Compute
columnwise, in float32 within each current network/layer:

    U = H - mean(H, axis=0)
    Z = clip(U / sqrt(mean(U*U, axis=0) + 1e-6), -8, 8)
    R = tanh(Z @ A)
    c[l,i,k] = c_star[l,k] + (1-abs(c_star[l,k])) * R[i,k].

A is one shared64x4 float32 matrix, initialized exactly zero, the ONLY
trainable parameter(256). Apply on layers1..14; no action at Gaussian
layer0 or terminal15. No bias, extra feature, layer head or architecture
sweep. Coefficients index BIRTH COLUMNS, so

    DeltaB[i,j] = sum_k residualized_F[k][i,j] * c[l,j,k].

Use the existing residualized F1..F4, original t/s/a, actual masked FP32
stored common base, stored-value plane increment and star restoration
unchanged. The existing scalar multiplication becomes column broadcasting,
with the SAME summation order. At A=0 it must be byte-exact to the successful
native parent, although the additional head work must still be paid.

H is already available before the current mean correction is applied.
Do not normalize using another net, cached teacher state, target, validation
statistic, or state computed after injecting the current target correction.
Freeze P parameters but retain derivatives through P/H and all endogenous
past state with respect to A. Discrete selections remain piecewise constant;
checkpoint replay must reproduce their physical memberships exactly.
No invocation-global mutable head state that can survive between networks.

## Fidelity before fitting

Write the exact implementation source/fixtures and pins before execution.
No estimator arithmetic, model import or job from an inert builder.
Require literal stored-pair and zero-head coefficient/broadcast checks,
including nonsymmetric column coefficients and CPU/MPS derivatives.
Use the fixed nonzero test matrix A[i,k]=float32(.01*sin((i+1)*(k+1)))
(offline definition, not fitted). Native null outputs must reproduce all16
parent output bytes on development35/100/35, initialization once, repeat
exact. Nonzero native/CPU/MPS full-size means must agree within1e-5,
final-MSE ratios within1%, and a non-negligible fixed-direction gradient
CPU/MPS discrepancy within5%. No cap is combined in this pilot.

No nonfinite prevariance/gradient/state, no prevariance below1e-10,
no checkpoint-selection mismatch. Repairs can version numerical code;
do not tune A, normalization, bounds or thresholds to rescue a gate.

## Small objective pilot, conditional on fidelity

TRAIN is ranks35..58 excluding37,39,42,48(20); VALID is37,39,42,48(4).
These are REUSED development networks, not pristine holdouts, and are
disjoint for the new A fit. Use existing authenticated1e9 mc_post means.
Six epochs, batch4 with immediate per-net backward, seed9309093,
Adam(lr.01 cosine to.001 over30 updates, betas.9/.999,eps1e-8),
no weight decay, gradient norm cap5. Loss is
(final MSE + .05*mean earlier-layer MSE)/2e-8. Initialize A=0 once.
No extension, restart, checkpoint selection or hyperparameter sweep.
Only the terminal A may be judged. Seal its payload and ALL4 validation
predictions before decoding those validation targets.

Native terminal predictions and COMPLETE bills must be acquired together;
compare to the existing qualified official-input birth-direction parent
on the same4 nets with matched initialization ownership. Also compare own
zero-head control. Do not pair new raw with an estimated added cost, clip
away raw increases, or substitute a tensor/derivative proxy for the score.
Advance only if mean actual adjusted improves by at least3% against BOTH
controls and at least3/4 nets improve against BOTH. These are pilot
admission gates, not an independent14 or official milestone.
Any larger fit requires a separate prospective protocol; no automatic
production build, panel, upload or response-row exception reopening.

## Resources and release

One MPS job total, at most6 nice10 numerical workers total, CPU BLAS1.
No AP+threadpoolctl or MPS fallback. One resident network graph;
training RSS<12GB, MPS driver<16GB, monitor swap. Native8GB/120s;
later NumPy-free production residual<.4s and actual packaging checks
remain required before a bank. Preserve small validated gains as
components; submissions only at the authorized<4/<3/<2e-9 milestones,
with the final official goal strictly below1e-9.
