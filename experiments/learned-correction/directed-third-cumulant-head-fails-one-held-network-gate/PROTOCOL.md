# Directed incoming-third correction on current autonomous inputs

Phase 2 only. Prospective after checkpoint91; official parent331684 is
facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0.
The upload-rejected, separately qualified covariance/readout archive stays
unchanged; this local experiment does not retry that upload.

## Distinct question and fixed model

Can incoming recurrent messages plus the current directed pair state predict
the OFF-DIAGONAL error in preactivation K21, before the pair map? The old
single-cell H65 bilinear fit failed next-mean information gates. The frozen
history covariance decoder failed its actual endpoint. Neither trained this
directed nonlinear pre-K21 model on current331684 autonomous inputs and the
public100M third planes. These negatives remain unchanged. Strong nonlinear
native feedback means a successful frozen-input fit is only a screening
result, never a final-score forecast.

Only layers8..14. Immediately before pair_map, use the PREVIOUS layer's
actual parent H(n,64), transported as W.T H and (W*W).T H. Current-layer H
cannot be used here: it is computed later and would create a circular
dependency. All future native corrections feed the ordinary pair map, edge
source, mean/BD heads, and coordinate-star repair; the diagonal of K21 stays
exactly unchanged. No post-third label is claimed: the public files supply
pre_M21 only. Existing CP factors are not separately rewritten.

The140 node columns, in order, are 1,l/15,alpha,gate,pdf,alpha*pdf,
(alpha^2-1)*pdf, log(sigma)-mean(log(sigma)), diag(K21)/sigma^3,
radial, mean(R,row),sqrt(mean(R^2,row)), W.T H64,(W^2).T H64.
R=Cpre/(sigma outer sigma), S=K21/(sigma_i^2 sigma_j). Node columns
are divided by global TRAIN RMS (floor1e-5), then clipped to[-8,8]. R and
S use TRAIN offdiagonal RMS floors1e-5. No per-network or target lookup.

One shared32-hidden directed head:

    z_ij = F_i left + F_j right + bias
           + R_ij edge0 + S_ij edge1 + S_ji edge2
    delta_S_ij = target_scale * dot(tanh(z_ij),output), i != j
    delta_K21_ij = sigma_i^2 sigma_j delta_S_ij.

The target is (central public pre_K21 - current K21)/(sigma_i^2 sigma_j),
zero diagonal. target_scale is one TRAIN offdiagonal RMS. FP32 deployed
messages, features and head, FP64 addition to K21. Process64 output rows
per block; no n*n*32 persistent buffer. Positive scales are frozen from
TRAIN only; no model-size, feature, layer, amplitude or precision search.

## Acquisition and numerical admission

Reuse the metadata-selected split sealed September6 in
public_p2_moments_metadata_v1.json: TRAIN57,68,45,59,3,12; HELDOUT14,71.
All are MINI ordinals, not FULL development ranks. Identities are disjoint
from FULL1000 and virgin6 by the prior metadata audit. Keep pinned commits,
sizes and SHA256 values. Only57's moment payload is already local.

First meter the fixed1024-wide head and qualify its equations without data.
Require a conservative seven-layer component bill including all32-bit
parameters cold to be <=15 billion weighted FLOPs. This is an admission
cap, not the complete final estimator bill. Then capture all eight current
autonomous input trajectories before admitting any new moments. Capture57
is bracketed by a plain-parent call and an observed repeat: require all16
output rows, full bills and repeated capture bytes exact. Passive taps only
retain existing objects inside the counted call, with no new arithmetic.
Other captures run once from a fresh cold module. Keep unfloored variances
positive, outputs finite and the parent's numerical cache retired.

Weight access projects only mlp_id/mlp_name/mlp_seed/weights from the pinned
local MINI Parquet cache. Freeze current input features and response maps
before target decoding. Public TRAIN downloads are serial and size/hash
bounded. HELDOUT moment files are not downloaded or decoded until the sole
trained model and all normalizers are frozen. gt_mean and public1B
mc_post labels are not decoded by any fit or information judge.

Allow only identity scalars, pre_mean/pre_m2/pre_m3/pre_M11/pre_M21, mean/M11
for100M centering and consistency. Reuse the certified directed centering
helper. For each network require original Gaussian layer0 mean/covariance
standardized RMS<=2, mean max<=12, third RMS<=1.5 and max<=12; raw diagonal
agreement<=2FP32ULPs. At each layer8..14 require affine mean RMS<=1e-6,
covariance relative<=1e-5, and stored-FP32 rounding-bound RMS<=5% of the
missing offdiagonal third-plane RMS. These checks exclude a storage or
identity failure; they do not remove late-layer sampling error or bias.
Attribute the higher-moments repository and retain the source-dataset
CC-BY4 attribution; its repository MIT declaration does not erase that.

## Fixed consumer loss, optimizer and information gates

Let A1=gate,A2=pdf,A3=-alpha*pdf,A4=(alpha^2-1)*pdf. With diag(R)=0,

    L_ij = sigma_i sigma_j/2 * sum(r=0..2)
           R_ij^r/r! * A_(r+2,i) * A_(r+1,j).
    C(D)=L*D+(L*D).T, diag(C)=0
    N(D)=pdf_next/(2*sigma_next) * diag(W_next.T C(D) W_next).

L uses the current frozen parent's sigma/alpha, and the next response uses
its actual next input. It is the real-arithmetic degree2 cubic contribution
to the pair covariance, not the full floating-point derivative or final
mean Jacobian. Preserve directed R as used by the effective consumer.
Set w=(L^2+(L.T)^2), zero diagonal, divided by its offdiagonal mean.

Training loss is .25*weighted-plane MSE + .25*C MSE + .50*N MSE. Each term
is divided by its zero-predictor TRAIN mean energy at that layer. Equal
weight for42 cells, all offdiagonal entries and all1024 response coordinates.
Loss/capture arrays use FP64; model forward/parameters use FP32. A blocked
NumPy backward implements the same loss; certify gradients in FP64 tiny
fixtures by central differences before wide training.

One model: rng seed712031, left/right normal SD .15/sqrt(140), edge normal
SD .15, bias/output zero. Adam24 epochs,42 full-cell steps per epoch;
fixed rng permutation per epoch, learning rate .002*.98^epoch, beta .9/.999,
epsilon1e-8, global gradient norm clipped to5. No regularization, averaging,
early selection or later rescue. Save final epoch24 only for judgment;
intermediate scalar TRAIN losses may be logged. Normalizers/denominators
come only from the six TRAIN files. Require all finite throughout.

After freeze, admit and qualify both heldout files. Report each consumer
separately, ratios of summed errors against zero prediction using fixed
TRAIN per-layer energy normalization, and individual layer ratios. Advance
only if EACH heldout network has weighted-third<=.85, covariance<=.85 and
next-mean<=.90. Failing any closes this model without more epochs, tuning,
native candidate or extra labels. This is deliberately a material-gain
screen relative to the added cost; it is not a universal information bound.

If it passes, prospective native implementation and qualification follow
with unchanged architecture, final payload and amplitude1. The initial
actual endpoint is FULL36/38, each arm's own complete cold bill and official
max(.1,F/2^41) multiplier, predictions frozen before1B labels. Require pooled
adjusted<1 and each<=1.05. Only then fixed14 panel40,41,43,44,45,46,47,49,
50,51,52,53,54,55 (<1,>=8wins,worst<=1.10), confirmation37,39,42,48,59,63,
66,67 (<1,>=5wins,worst<=1.10), client/runtime/contract/package checks.
No upload is selected by this protocol. Official~1e-9 remains unmet.

## Resource and preservation contract

One nice10 CPU group at a time, six BLAS variables1,8,000,000,000B tree-RSS
limit,256MiB emergency swap cutoff and ZERO added swap for qualification.
No GPU or sampling. Monitor .5s and retire each process group. Download
<=2,356,640,712 new bytes (seven pinned files); total new artifacts<=8GB;
require>=480GB free before acquisition and >=470GB throughout. Ten native
parent calls maximum: three57 qualification calls plus the seven others.
Guards: units180s, each capture group900s, each download300s, each cell
preparation600s, training5400s, heldout judge600s. Whole fitting stage may
fail its wall cap; do not reset it for more training. Bounded implementation
repairs preserve every original source/failure and cannot change this design.
Seal each stage's executable sources before it runs.
