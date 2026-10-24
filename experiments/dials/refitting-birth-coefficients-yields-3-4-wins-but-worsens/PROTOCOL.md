# Current-COMP bounded-headroom cstar proposal

Prospective Phase2 experiment,2026-09-10. This is a NEW approximate-proposal
experiment, not a repair, continuation or numerical qualification of
current_comp_cstar_information_v1 (CS). Root must review all source and
separately authorize each launch. No numerical execution follows from source
preparation. Current implementation scope initially comprises this protocol
and inert model; separately reviewed worker/supervisor owners will implement
the fixed stages below without changing these scientific choices.

## Evidence, purpose and preserved failure

Original56-active-parameter BD fitting achieved5.18564% native40 raw and
4.89372% adjusted improvement,38/40 wins, then both independent14 groups.
P1 and NODE subsequently changed the conditional dynamics while cstar stayed
fixed. No executed current-COMP cstar fit was located. P1's explicit
approximate-proposal/native-objective process subsequently earned.81955%
native adjusted improvement,29/40 wins, and both independent-panel passes.
That is process evidence, not a cstar effect-size forecast.

Expected gain is UNKNOWN. No stationarity claim follows from old changing-
iterate loss histories; no useful remaining direction follows from bounded
coefficients alone. The unresolved premise is that a fixed approximate
optimizer finds motion that transfers to the actual native endpoint.

CS qualification remains FAILED: rank35 MPS h=.025 scalar secant discrepancy
.2999523038318002 exceeded the unchanged.05 gate after22 retired calls.
Worker result d2c01e7013633c23b0fe97ce0a2444065232527ab8730951ee7eb432df36a77b
and watch adb4a133dc06d21164d766a4e9e9a5a93867a7a6bd1456e0b8e833e9591fd555,
their completion/finalization chains, source and parked information drafts
are immutable failure evidence. Neither partial CPU controls nor the old
information admission grants this experiment eligibility. There is no new
Gline, scalar rescore, finite-difference amplitude, FAD/RAD rescue or lowered
CS threshold. CO, RB and all other old outcomes also remain unchanged.

## Exact parameter surface and numerical supplier

Parent is retained COMP
2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a:
P1 P, retained NODE A, original exported FP32 c0[15,4]. No D6, receiver,
covariance or common-cubic intervention is composed here. All P fields,
normalizers, inactive66 P coordinates, NODE A, weights and c0 bytes remain
frozen. H and every physical/CP/star/radial recurrence remain live.

Train exactly one leaf delta[14,4],56 allocated/active entries, initialized
to positive FP32 zero. In the stated FP32 operation order compute

    headroom = ones_like(c0[1:]) - abs(c0[1:])
    active_c = c0[1:] + headroom * tanh(delta)
    actual_c = concat(c0[:1], active_c).

There is no inverse tanh, retanh of c0, clipping, straight-through custom
gradient or zero-action bypass. All active original c0 entries are nonzero;
the computed zero-delta tensor must reproduce every c0 byte. Row0 is copied
by concatenation and has no parameter. In real arithmetic, finite delta
keeps c inside(-1,1), with relative displacement bounded by1-abs(c0).
FP32 tanh/addition can reach a boundary; the actual stored tensor must be
finite with abs(c)<1 or the stage FAILS without clamp, LR change or retry.

This is a NEW relative-headroom coordinate system. At delta=0 its diagonal
Jacobian is1-abs(c0), not old theta's1-c0^2. The selected learning rates
therefore are not described as theta-equivalent or evidence of an optimum.
The original NODE rule c+(1-abs(c))*tanh(Z(H)@A) remains unchanged, including
its c-dependent amplitude and endogenous H derivative. The actual nonleaf
coefficient tensor reaches the existing CS card by identity; no detach or
copy cuts the delta graph. All physical steps/checkpoints are inherited.

Model API: authenticate()->stdlib pins; runtime()->original BD base;
frozen(device,trainable_delta=True)->P,A,delta; coefficients(delta)->actual
FP32 c[15,4]; card(rank,*,P,A,delta,device='mps',weights=None);
reference_card(...) is unmodified same-backend COMP; _CO/PKEYS/INACTIVE_KEYS
and source_proof() remain exposed. card.delta/c0/headroom/coefficients
identify all new graph/constant owners. card_from_coefficients(...,cstar=...)
is the separate nontrainable terminal-validation path, using the actual
sealed exported tensor, never an inverse reconstruction of delta.

## Stage1: five fresh target-free execution-integrity calls

One fresh MPS process. Roster: unmodified COMP reference35, reference100,
new delta-card checkpointed backwards35,100,35. Both references are fresh;
no failed CS output is used as a successful control. Use original CO
synthetic Gaussian cotangents from seed9309098+rank and its original scalar
functional/physical-MPS arithmetic, unchanged. No truths or model updates.

One persistent actual delta leaf and Adam object. After each backward,
execute Adam at exactly ZERO LR; require finite allocated step/exp_avg/
exp_avg_sq state for the sole56-value leaf, unchanged positive-zero delta
bytes and unchanged exported c0. Require finite full56 gradients and
nonzero full gradient norm, without a native-gradient assertion. First and
repeat35 outputs/scalars/full gradients and all applicable diagnostics must
be byte-identical. Every delta-card null raw/served output, variance-domain
and selector bytes must equal its corresponding unmodified reference.

Require16 finite unfloored prevariances>=1e-10,47 original selector cells,
all14 NODE layer keys and byte-exact every observed checkpoint replay.
Require actual max replay visits>=2 after reverse, not a fabricated minimum
over early-stopped layers. Frozen P/A/c0/headroom/weights remain unchanged;
all graph/card/output/supplied-nonleaf/delta owners retire before a call is
marked complete (the one persistent optimizer delta is retired at stage
end, after the final allocation check). No capture, extra model or JVP call.

PASS means execution integrity and resource qualification ONLY. It does not
assert native derivatives, CPU/MPS scalar fidelity or satisfaction of CS's
failed gate. Any numerical/domain failure closes this version. Only source-
plumbing failures may receive a separately reviewed new version, retaining
the original source and failed receipts. All5 completed calls and worker/
watch result,completion,finalization seals are required before Stage2.

## Stage2: one TRAIN160 pass and sole terminal

Use the existing fixed160 TRAIN /40 VALID roster from
joint_prefix_final_data_v1.py
22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b.
These are reused research networks, not pristine holdouts. Authenticate
canonical weights and documented existing1e9 TRAIN post means before their
use. No CHECK4/VALID/independent14 target is accessed during proposal work.

One PCG64DXSM permutation, seed9309099. Four nets per update, immediate
per-net backward/retirement,160 TRAIN graphs and exactly40 fresh Adam
updates. No optimizer resume. Learning rate at update u=0..39 is
.0025+.5*(.025-.0025)*(1+cos(pi*u/39)); betas(.9,.999),eps1e-8,
weight_decay0,foreach=False and global gradient-norm clipping5. No epoch,
LR, coordinate, seed, amplitude, feature, checkpoint or radius selection.
Every post-update delta/export/domain must be finite; strict coefficient
bounds and frozen constants apply throughout. No post-hoc learning-rate or
domain rescue. Record clipping and saturation only as diagnostics.

Reuse P1's exact fixed cold d50 cost weights from TL capture manifest
cb16019fe0a72b1ef11991e253992d3a9544d8f1b28df403b2c8b2ee290a650a:
a_i=max(.1,F_i/2^41), normalized by the TRAIN mean. These are PROPOSAL
CONSTANTS, not current COMP or proposed-candidate bills/compute derivatives.
Loss per net is a_i*(served final MSE+.05*served earlier15 mean MSE)/2e-8,
then averaged over each batch4. Preserve P1's finite stored target dtype
and explicit FP32 target cast for proposal loss. Report online raw and
weighted/unweighted surrogate statistics; none is acceptance evidence.

After update40, seal exactly one terminal delta and the actual computed
FP32 c[15,4], with c0/P/A hashes, source/input lineage and40-update/160-graph
proof. Only c, not delta or tanh, enters any subsequent native estimator.
Retire training graphs/optimizer and reload the sealed c as a nontrainable
tensor; acquire exactly40 target-blind surrogate VALID forwards through
card_from_coefficients. Seal all40 outputs before any validation truth.
Exactly200 complete model calls,160 backward+40 forward, and one terminal.
No acceptance, step selection or early termination based on surrogate loss.

## Native objective, source proof and advancement

After separate source review, construct exactly one candidate by replacing
ONLY the original _BD_COEFFICIENT_LITERAL value span with sealed FP32 c.
Prove all other COMP source bytes unchanged and literal roundtrip byte exact;
null source is original COMP. No new online parameter conversion, heads,
bases, factors, current-plane terms or meter mutation. Reuse original
official-owned input/reset/bill/lifecycle semantics and correct nested
P/cstar/NODE-A initialization ownership. Every changed candidate pays its
OWN complete bills; selector/width changes are measured, not assumed absent.

First fixed CHECK4 ranks37,39,42,48, plus repeat37: seal all outputs/bills/
domains before those original1e9 truths. Advance only if mean per-net
raw_i*max(.1,own F_i/2^41) is strictly below COMP and at least3/4 strict
per-net adjusted wins. No fit/checkpoint/step selection follows this screen.
Then unchanged terminal on full original40 plus repeat37, requiring strict
mean adjusted improvement against COMP and at least24/40 strict wins. Then
strict NC-adjusted improvement on BOTH original public8 and virgin6 groups,
with their original unclipped noise subtraction and own complete costs.

No native/surrogate gradient or fidelity claim follows from an endpoint
gain. Current COMP mean468313646101.175 F is planning context only; a
same-shaped literal replacement adds zero operations at fixed geometry.
No1.7e-8,sub3 or milestone gain is forecast. Production NumPy-free/runtime/
memory/source/package gates and the original milestone-only bank authority
remain separate. No automatic submission, root estimator edit or promotion.

## Resources, artifacts and launch boundary

Separate explicit launches: qualification300s and proposal1800s, each whole
clock includes source admission, every model call, rehashing, all3 seals
and owned-group cleanup. One nice10 AP child, bytecode disabled, all five
BLAS variables1, Torch intra/inter-op1, fallback disabled, no other numerical
worker. NumPy1.26.4/Torch2.7.0; NEVER import threadpoolctl. MPS low watermark
.1 and allocator fraction min(1,16e9/recommended), configured before tensors.

External combined supervisor/owned-tree RSS<12GB, zero added system swap,
actual completed sampling intervals<=1s, sampled MPS driver<16GB. Host and
driver overlap on unified memory; these are not summed or claimed as an
absolute system peak bound. No interval>120s without the next ordered fully
checked/retired graph, including admission and post-final-call sealing.
Check deadlines before accepting progress; no deadline resets on heartbeats.
Use the reviewed absolute-deadline monitor and retire experiment plus
observational process groups on success/failure. No CPU/MPS overlap.

Exclusive outputs: qualification_v1 and training_v1, corresponding
qualification_watch_v1 and training_watch_v1. Every worker/watch publishes
actual result.json,completion.json,finalization.json; no PASS before complete
publication/reap verification inside its whole clock. Flat source/input/
weight/target/artifact maps, immutable ordered call records and actual
resource observations are required. Preserve every attempted artifact.
Later owners must expose explicit source/model/protocol/watch/spec hashes;
training additionally pins fresh qualification and watch result digests and
verifies both complete3-seal chains. No placeholder may authorize execution.

P1's approximately12-minute200-call stage is only timing precedent. Five
current-cstar calls have not been timed under this new parameterization.
No help, import, self-test, model call, fitting or numerical preflight is
authorized by this source preparation; root reviews and selects launches.
