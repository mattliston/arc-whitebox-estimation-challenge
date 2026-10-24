# Warm-started observer on the fixed accepted-Lazy trajectory

Phase2 only, selected2026-09-20 before new capture or training. Parent source
b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c,
official#331599 adjusted3.8164492657795035e-9. The~1e-9 objective and
qualified-bank preauthorization remain unchanged.

## Why this exact experiment

Raw-third recentering was actually tested (one-step ratios1.0894/1.1820),
and translation-consistent feedback and its matched fits already ran. Neither
is rediscovered as a bug fix. The current covariance-tangent additive65
readout also completed negative:1.025275 adjusted and1.000160 even at zero
added cost,0/5 fold wins. Keep all those negatives unchanged.

Read-only recurrent observers are not a new general family. The old
current_v29_temporal_observer_v1 trained a random fresh observer on a
different public fourth-order physical host and obtained TRAIN ratio.980454,
failing its.90 gate after150 updates. Do not reuse or extend that endpoint.
The older joint-prefix/final fit changed physical feedback and achieved only
.998434 on40 validation networks; its negative also remains unchanged.

Here freeze the ENTIRE actual accepted-Lazy physical trajectory, including
its original P, NODE, cstar, all means/covariances/tensors, selectors, weights
and original GNN state. A separate neural recurrence starts from a COPY of
that current Phase2 P and consumes its recorded30 local features and weights.
Only its final readout replaces the original final GNN correction. Its hidden
state/corrections NEVER enter the original physical recurrence or birth
coefficients. This removes train/deployment physical-feedback mismatch for
the additional observer. It is a selected different host, initialization and
consumer, not a retrospective pass of either old observer/fit.

## Fixed model

Use the exact current gnn_step equations: H0=0, no encoder at layer0;
layers1..15 standardize original30-feature x by frozen P centers/scales,
clip to[-8,8], and apply the original15 mask. Then64->48 message projection,
three16-channel W/W^2/W^3 messages and their existing squared-message
energies/log1p, standardized-gate multiplication, message clip,126->128
SiLU->128 SiLU->64 encoder. Final correction retains the original
(FP32 sigma*rms15)*(H15@head15+head_b15)*(1+.25*tanh(gain15)).

The proposed final mean is max(original FP64 scalar final host + new FP32
neural correction,0), servedFP32. Earlier15 rows remain the original parent.
The original final GNN may still be computed; no omitted-cost credit is taken.
Only eight message/encoder arrays and head15,bias15,gain15 are trainable:
msg_w,msg_b,enc0_w,enc0_b,enc2_w,enc2_b,enc4_w,enc4_b,head,head_b,gain,
where the last three are final-row leaves. All other P values are frozen.
Initialize every trainable/frozen value exactly from current accepted P.
No old fitted observer, Phase1 value, random trunk or fitted normalizer.

No architecture/width/feature/layer/rank/precision search, ensemble, auxiliary
loss, teacher feedback or new physical sample. Expected improvement unknown;
this is a possible component, never a claim that~1e-9 is reached.

## Acquisition and qualification before targets

Passively wrap original gnn_step to retain the existing x,sigma,hidden and
delta references at all16 layers, and original nonlinear_scalar to retain
its final FP64 scalar host. Add no numerical operations inside the counted
parent. After inference, copy small arrays for offline use. Verify the final
host+native-delta reconstruction exactly; every stored value finite.
Preserve all16 output bytes, complete bills, weights and cache retirement.
The original source is not edited and installed numerical libraries stay
unchanged. Store no new dense weights/covariance/carrier tapes.

First run native35/100/35 in one cold/warm/repeat lifecycle, matching the
saved Lazy controls exactly. All repeated capture arrays and P must agree.
Recompute the copied observer with original FNP gnn_step on the saved inputs;
require every hidden/delta and served output byte exact. Measure its complete
neural replay work with actual FNP, including input/P preparation; require
each added-neural replay bill<=8GF. This is not full deployment qualification.

Before training, qualify the explicit Torch model against these native rows
on CPU and MPS: final served maxabs<=1e-6 and RMS<=1e-7 at original P and
at ONE fixed enc4_w direction, plus/minus2^-8, where direction_ij=
sin(.17*(i+1))*cos(.11*(j+1))/sqrt(64*128). Use actual FNP neural replays
as references, no new physical trajectories. Tiny CPU64 central-difference
checks (epsilon1e-5,relative<=1e-5) must exercise all eleven parameter paths.
On native35's fixed feature trajectory, compare CPU/MPS gradients of the
fixed scalar dot(final neural delta,sin(.13*(j+1))/sqrt(1024)); all finite,
nonzero total norm and relative difference<=.01. No targets in qualification,
no claim of a physical/backbone gradient, no direction/tolerance rescue.

Only a qualification pass acquires the remaining19 TRAIN and4 CHECK rows,
each independent cold process, exact all16 outputs and bills against the
existing accepted-Lazy receipts. TRAIN20:35,36,38,40,41,43,44,45,46,47,
49,50,51,52,53,54,55,56,57,58. CHECK4:37,39,42,48. Reuse qualifiedcold35;
qualification100 is not a training or scoring row. All24 feature sets sealed
before any target is decoded. These are repeatedly used development networks,
not pristine holdouts. Use only the existing original1B mean references;
the protected observer688/audit/lockbox target bundles are not involved.

## One fixed fit, endpoint and score law

Thirty epochs, twenty TRAIN networks per epoch, fixed permutations from
PCG64DXSM2026092009, batch4, immediate per-network backward divided by4:
600 backwards and150 optimizer steps. Final mean loss only, normalized by
2e-8 and weighted by each network's complete cold multiplier divided by
the TRAIN mean multiplier. Targets castFP32 for MPS; final reports use
their original stored values widenedFP64. The scalar host is castFP32 for
MPS; qualification bounds its discrepancy. No loss on intermediate layers,
hidden states, covariance, cost derivative or CHECK networks.

AdamW, betas(.9,.999),eps1e-8,weight_decay2e-5,foreach=False. Global
gradient norm cap5. For zero-based step t0..149 define c=(1+cos(pi*t/149))/2.
Eight trunk arrays use LR=5e-6+(5e-5-5e-6)*c; final head,bias,gain use
LR=5e-5+(5e-4-5e-5)*c. All initial values from exact P; no random model
initialization. Frozen fields remain byte-identical. No checkpoint selection,
epoch extension, optimizer/seed/LR/radius sweep, head polish or resume.

Seal the ONE terminal payload after step150. Retire optimizer/live graphs,
reload it nontrainably and produce all20 TRAIN plus4 CHECK cached predictions
without providing targets to forward(). Seal predictions before scoring.
First open only TRAIN values and require mean(own raw*own multiplier), using
parentF+8GF for observer and parentF for baseline, ratio<=.95. If that fails,
stop without CHECK labels or a native candidate. Only then open CHECK4 and
require pooled own-bill ratio<=.97,at least3/4 wins,maxindividual<=1.02.
The8GF is fixed provisional incremental allowance, not an actual deployed
bill. A passing endpoint still needs a separately sealed actual native port,
complete billing<=allowance, all repeats/contract/cache checks, fixed NC14,
confirmation8 and actual client<=45s and<=1.10 accepted-Lazy pooled wall.
Only then package/upload under existing preauthorization. Failure leaves
no continuation/parameter/consumer rescue within this selected study.

## Resources and execution staging

One numerical group at a time,nice10,six BLAS thread variables1,host RSS8GB,
zero added swap; existing256MiB emergency swap stop. MPS hard allocation
limit8GB,lowwatermark.1,fallbackdisabled; graphs retired between networks.
Use current OpenBLAS/FNP runtime for native and AP NumPy1.26.4/Torch2.7.0
for neural work. No simultaneous FNP/MPS workers or all-network dense powers.
Capture qualification180s; FNP replay60s; CPU/MPS qualification180s; each
new physical capture100s; serial capture controller2400s; fixed fit900s.
At most220MB new artifacts. No dense weights copied to new files.

Stage sources must be sealed before their numerical execution. Capture and
FNP replay can be implemented/sealed first; neural/model/trainer sources must
be sealed before their own qualification and ALL before targets. Later code
completion may not change the algorithm, roster, optimizer or gates above.
Preserve failures and numerical receipts/version history; no hidden retry.
