# Current terminal bilinear history feature V3: signed cold diagonal

Phase2,2026-09-20 UTC. Written before new arrays or targets are decoded.
Parent is exact submitted Lazy,b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c.
Official#331599 remains3.8164492657795035e-9 adjusted; approximately1e-9 goal
active. Qualified future uploads preauthorized; no upload in this study.

## Overlap and distinct question

This is a RELATED dictionary to the CLOSED learned endpoint-history covariance
decoder, not a new general model family. That old six-network fit used P0 on
coherent8M teacher trajectories, three R^0/R/R^2 bilinear kernels shared over
layers8..14, and a joint covariance/next-mean objective. Its held next-mean
ratios.9933264149/.9855317053 failed. The old H15 quadratic, extra382 features,
bagging, two-hop and terminal diagonal-C65 tests remain negative too.

The present question is narrower and directly addresses the task's objective:
can a single final-layer R^0 bilinear history read predict the current autonomous
Lazy final mean residual across networks? Use current H14/C14 and exact final
W, a closed-form TRAIN-only solve against ordinary1B final means, and a matched
existing-head control. No teacher covariance labels, recursive writes, P0
payload transfer, Adam retry or claim that the old decoder was defective.

## Fixed feature and arithmetic

Let U=(sqrt(where(alive,diag(C14),0))[:,None]*H14) roundedFP32. Require
all covariance diagonals finite and ACTIVE diagonals strictly positive;
retain signed cold diagonals literally. Require final pre-v>=1e-10.
No variance floor/repair. Existing alive is the post14 mask. Let Z=W15.T@U,
using FP32 products. For all2080 upper-triangular pairs a<=b, compute

  F_ab = phi(mu15/sigma15)/(2*sigma15) * (2 if a<b else1)
         * [Z_a*Z_b + (W_cold^2).T@(Cdiag_cold*H_cold,a*H_cold,b)].

Active upper-pair products and multiplicities are FP32. Form the signed
cold Cdiag*H_a*H_b product in FP64 and roundFP32 before its multiplicity and
FP32 cold matrix product. This algebraically avoids undefined cold square
roots; it preserves the real positive-domain formula but changes its rounding.
V2 failed the stricter domain and remains failed (see SIGNED_COLD_V3.md).
The scalar
Gaussian variance-response factor and its final multiplication use FP64,
then features roundFP32. Compute sigma/pdf uniformly from captured mu/v;
no later tensor, H15 or target supplies this feature. These2080 columns are
the Gaussian variance read of a signed bilinear low-rank covariance ansatz
with existing active/cold-diagonal ownership. The proposed consumer is an
ADDITIVE final-mean residual readout after the unchanged estimator. It is
not a true covariance supplier, full final-GNN covariance response or joint law.

The existing final GNN compresses H14 to48 message channels, then three
16-channel edge-power blocks. Full64-channel first-power queries can therefore
carry additional interface information, but no residual value is assumed.
All2080 columns are fixed; no feature or layer selection after labels.

## Target-free qualification and capture

First test actual FNP feature contractions against independently materialized
small signed covariance ansatz matrices, including active/cold ownership and
empty/full active sets and signed cold diagonals; relative error<=3e-6. Test
strict ACTIVE-domain/nonfinite rejection and
the existing weighted affine/extension projection/solve certificate unchanged.
Unit worker60s. All initialization, casts, scans and feature work metered.

Then read only the sealed TCI35/100/35 capture packets. They already proved
exact all16 parent bytes, full bills and repeat arrays. Authenticate source,
weights and every receipt/guard. Compute features in separate counted contexts,
require exact repeated feature arrays and bills, finite1024x2080 FP32 outputs,
each feature bill<5GF. Seal them before targets. Reuse only cold35 for TRAIN;
100 is qualification-only. This does not assert a new full native prediction.

For the other19 TRAIN networks, passively wrap the unchanged parent's final
covariance/scalar/GNN helpers to retain C/H14/alive/W/mu/v/H15/sigma. No new
numeric operations inside its context. Each native prediction starts cold,
must reproduce ALL16 bytes and full bill from that rank's previously sealed
exact-Lazy two-hop capture, and must retire all W5 numerical cache state.
Then compute/meter the same features separately. Store only features,H15,
sigma,prediction and scalar receipts; do not store new C/H14/weight tapes.

Fixed TRAIN20:35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58.
No CHECK37/39/42/48 truth access. Seal every capture/receipt/weight/source
before opening TRAIN references. Target manifest is the preserved joint-prefix
fit manifestd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971.
These are reused research networks seen by the backbone; results must be
called exploratory TRAIN-network cross-validation, never fresh generalization.

## One fixed value test

Use the exact weighted projection/ridge functions from
terminal_representation_v1/information.py. Five folds by TRAIN ordinal modulo5;
16 fit/4 held networks. Control B=sigma15*[1,H15]. Extension F above. Fit-only
weighted column RMS floors1e-12; project normalized F off the weighted B span
with relative singular cutoff1e-7, discard residual-column RMS<=1e-6, normalize
retained residuals and solve one ridge1e-4. No normalizer, ridge, cutoff, rank,
solver, roster or checkpoint search. Preserve fold objects and seal all held
FP32 nonnegative predictions before aggregate scores. Report equation residuals
and retain the original projection orthogonality<1e-5 guard.

Each sample's fit weight uses that network's OWN complete cold parent scoring
multiplier. Score mean_i(raw_i*max(.1,own_F_i/2^41)). Unchanged costs parentF;
affine parentF+2MF; extended parentF+ACTUAL metered full2080-feature bill+20MF
readout reserve. This is a conservative information-test charge, not a qualified
production bill: a future compiled64x64 readout could avoid2080-column materialization
but receives no discount here. Ordinary original1B FP32 means widened for MSE.

Advance only if pooled extended/unchanged<=.95, extended/control<=.97 and at
least4/5 held fold means beat BOTH comparators. Failure closes this fixed
current-final bilinear read: no fit expansion, ridge/feature/precision/consumer
rescue or new archive. A pass only licenses separately defining a full TRAIN
fit, actual native CHECK, NC14/confirmation/client/package checks before banking.

## Resources

One numerical process group at a time, nice10, all BLAS threads1,8GB group RSS,
zero added swap. Existing supervisor has256MiB emergency swap-growth cutoff.
Qualification60s; each of19 native captures100s; serial acquisition1800s wall
guard in the orchestrator; value fit300s. At most260MB new artifacts. No GPU,
new physical samples, dataset downloads, source estimator edits or external
messages. Preserve failed acquisitions and reuse only positively sealed rows.
