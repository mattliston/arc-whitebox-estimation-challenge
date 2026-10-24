# Joint causal incoming third/fourth pair head

Phase2 only, prospective after checkpoint92. Exact parent331684 source:
facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0.
The previous incoming-third model, its failed joint validation gate and
parameters stay closed. This experiment trains a NEW three-output head
from scratch on K21, K31 and K22 residuals. No learned parameters transfer.
The separate covariance/readout archive remains upload-blocked; no upload
is selected here.

## Evidence and distinction

The older coherent-input rank35 attribution gave covariance/next-response
ratios.209/.328 for offdiagonal thirds,.902/.844 for all fourths, and
.021/.021 jointly. These are older privileged-source results, not a current
score forecast; the second network's qualification never completed. The
old14-feature fourth-plane bilinear span and anisotropic trace update also
failed. None of those is this jointly supervised nonlinear causal head on
the current bank. Checkpoint92's single-third head reduced covariance
response error about28.6%, but failed its fixed two-network joint gates.

The current hypothesis is that third and fourth corrections must be
predicted together because their pair responses can cancel. The primary
information objective is their SUM through the actual directed cubic and
quartic consumer. Individual-plane auxiliary losses help constrain the
three supplies; their errors are reported without pretending they are
separate final-score objectives. This changes the scientific target,
not the old model's gate or stopping decision. A successful proxy must
advance to a complete native objective test to establish any actual value.

## One model and causal placement

Only layers8..14. Reuse the exact current-parent input captures from
incoming_third_pair_head_v1, with all eight saved predictions and cold bills
qualified there. Reuse its RAW140 features:12 physical node scalars and
previous-parent H64 transported by W and W squared. H is the previous
layer's state; the current layer's H is unavailable before its pair map.
R and S are the current correlation and standardized directed third plane.

One32-hidden tanh pair head has shared node left/right matrices,3 edge
coefficient rows for R,S,S.T, bias32 and output(32,3). One global TRAIN RMS
normalizer for each input feature and R,S (floor1e-5); clip normalized node
features to[-8,8]. One positive TRAIN RMS target scale for each output.
Use64-row blocks. FP32 messages/features/hidden/outputs, with FP64 physical
moment additions. The output columns are deltaT21,deltaT31,deltaT22;
symmetrize only the predicted deltaT22, then set all three diagonals to0.
Do not symmetrize any parent covariance, correlation or target.

At the pre-pair boundary:

    K21 += v_i sigma_j deltaT21
    k22 += v_i v_j deltaT22

Inside the effective _nonlinear_pair_batch, add deltaT31 to its normalized
T31 before the original FP32 series cast. All other pair channels, post
third/fourth targets, radial update, edge/mean/BD consumers and coordinate-
star repair then use those current predictions. The current marginal
K3/K4 inputs stay fixed at that layer. No new full-tensor carrier or exact
all-distinct fourth-to-third emission is claimed. Existing CP/birth/trace
approximations remain; changed autonomous feedback must be measured.

## Targets and new fourth-moment qualification

TRAIN MINI57,68,45,59,3,12; validation14,71. These are the same metadata-
fixed roles.14/71 were already used in checkpoint92 and are NOT pristine
new holdouts. Their fourth-moment fields remain unopened until this model
is frozen. No new downloads: all eight hash-pinned100M payloads are local.
No1B gt_mean or original score label columns are decoded here.

Add only pre_m4,pre_M31,pre_M22 to the previously admitted preactivation
fields. The first index of M31 is cubed. Center and subtract Wick products
using the SAME100M means/second/third moments. Let Ctruth=M11-mu outer mu,
vtruth=m2-mu^2 (use exact scalar diagonals). Then

    M31c = M31 - m3_i mu_j - 3 mu_i M21
           + 3 mu_i mu_j m2_i + 3 mu_i^2 M11 - 3 mu_i^3 mu_j
    K31 = M31c - 3 vtruth_i Ctruth_ij
    M22c = M22 - 2 mu_j M21 - 2 mu_i M21.T
           + mu_j^2 m2_i + mu_i^2 m2_j
           + 4 mu_i mu_j M11 - 3 mu_i^2 mu_j^2
    K22 = M22c - vtruth_i vtruth_j - 2 Ctruth_ij^2.

For CURRENT parent v,sigma,C,K21,radial, form exact source-order baselines
k4=radial*v*v;T31=(k4/(v*v))_i*R;T22=[radial*(v outer v+2*C*C)/3]/(v outer v).
Targets are trueK21/(v_i sigma_j)-parentK21/(v_i sigma_j),
trueK31/(sigma_i^3 sigma_j)-T31, and trueK22/(v_i v_j)-T22. Zero diagonals.
The baseline T22 may have tiny asymmetry; that target is not repaired.
The first target must reproduce the old frozen third target after its
declared sigma^2/v unit conversion within1e-11 relative.

Before labels, certify centering against directly centered non-Gaussian
toy samples, relative<=5e-11, and certify FP32 storage-error bounds. For
each file retain all checkpoint92 identity/lower/third qualification. New
M31/M22 diagonals must agree with scalar raw m4 within2FP32ULPs. At the
Gaussian first layer, use influence variances

    Var(K31hat)=(6 vi^3 vj+18 vi^2 Cij^2)/N
    Var(K22hat)=(4 vi^2 vj^2+16 vi vj Cij^2+4 Cij^4)/N.

Require standardized RMS<=1.5 and max<=12 for both. Certify these polynomials
by enumerating the24 Wick cross pairings. At each late layer require the
stored-FP32 rounding-bound RMS<=5% of EACH missing standardized fourth
plane. These checks do not remove late-layer MC error or shared arithmetic
bias. Preserve all sources and source-dataset attribution.

## Exact directed joint consumer and fixed loss

Use actual sigma,alpha,pdf,gate; A1=g,A2=p,A3=-alpha*p,
A4=(alpha^2-1)*p,A5=-(alpha^3-3alpha)*p. Every term at(i,j) uses rho_ij:

    C(D)=sigma_i sigma_j sum_(r=0..2) rho_ij^r/r! * [
      (D21_ij A_(r+2,i) A_(r+1,j)
       +D21_ji A_(r+1,i) A_(r+2,j))/2
      +(D31_ij A_(r+3,i) A_(r+1,j)
       +D31_ji A_(r+1,i) A_(r+3,j))/6
      +D22_ij A_(r+2,i) A_(r+2,j)/4 ].

Zero output diagonal. No half-plus-transpose substitution. This is the
real-arithmetic response of the effective source, not an asserted native
floating-point Jacobian. N(D)=pdf_next/(2*sigma_next)*diag(Wnext.T C(D) Wnext).
Store the two directed coefficients for21 and31 and the coefficient for22.
Plane weights: left21^2+(right21.T)^2; similarly31;
for22, .5*(coef22^2+(coef22.T)^2). Normalize each weight to offdiagonal mean1.

Five squared losses: the three weighted planes, joint C, joint N. Normalize
each by its zero-predictor TRAIN mean energy at that layer. Equal42-cell
weight; offdiagonal plane/C MSE and all1024 N entries. Combine with fixed
weights[.10,.10,.10,.25,.45]. Certify source-response formulas on symmetric,
asymmetric and active-floor tiny fixtures to2e-11, and all-parameter FP64
directional gradients to3e-6 before wide work.

One optimizer: rng seed923071; left/right SD.15/sqrt(140), edge SD.15,
bias/output zero. Adam32 epochs/1344 full-cell updates; fixed rng permutation
per epoch, lr.002*.98^epoch, betas.9/.999,epsilon1e-8, global gradient norm
clip5. FP64 loss/grad/moment accumulation; FP32 parameters/forward. Final
epoch32 is the sole model. No averaging, early selection, extra epochs,
feature/rank/amplitude search or reuse of the closed third model.

Meter one1024-wide complete head/correction component, including cold
parameter ingress; require conservative seven-layer bill<=15BF before
target admission. After final model/normalizer freeze, qualify validation
fourth fields and judge. EACH validation network needs joint covariance
ratio<=.90 and joint next-mean ratio<=.90 against zero correction, using
fixed TRAIN per-layer energy weights. Report all plane/layer errors. A
failure closes this fixed model. No native candidate follows a failed gate.

## Actual objective if the information gate passes

Implement that frozen amplitude1 model in the exact parent, with source
inverses for disabled mode and a legal FNP-only static payload. Verify
all exported parameter bytes, independent blocked FP32/FP64 equations,
permutation, zero-output, repeat and all joint pair-map output channels.
Require disabled parent prediction bytes and full bills exact. Require
enabled first8 output rows exact to parent, own repeats and warm/cold
outputs exact, positive unfloored variances, finite values and retired
numeric caches. No new clipping or PSD repair.

Initial native qualification/development: disabled35/100/35, enabled36/38/36
and separate cold38, seven full calls. Reuse the saved exact-parent controls
with matching cold/warm states; do not substitute mean costs. Freeze
predictions/bills before1B score labels. Require pooled adjusted ratio<1
and each<=1.05 using each actual max(.1,F/2^41). Independently verify MSE.
Then fixed14 panel40,41,43,44,45,46,47,49,50,51,52,53,54,55 needs pooled<1,
>=8wins,worst<=1.10; confirmation37,39,42,48,59,63,66,67 needs pooled<1,
>=5wins,worst<=1.10. Actual persistent client/runtime/contract/package and
exact external authorization remain necessary. No proxy score earns a bank.

## Resources and preservation

One nice10 CPU group, six BLAS variables1,8,000,000,000B tree-RSS ceiling,
256MiB emergency swap cutoff and ZERO added swap for qualification. No GPU,
sampling, downloads or dataset copies. Reuse all captured features/weights.
New information-stage artifacts<=2.5GB; require>=480GB free initially,
>=470GB throughout. Guards: units180s; one-network target preparation600s;
training5400s; validation600s; independent audit600s; any native sequence900s.
Seal executable stages before running. Preserve original errors and additive
implementation repairs without changing this design or its gates.
