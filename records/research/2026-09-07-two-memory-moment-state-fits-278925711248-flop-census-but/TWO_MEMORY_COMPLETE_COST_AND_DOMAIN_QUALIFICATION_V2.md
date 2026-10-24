# Two-memory completion: conditional cost fits, domain issue unresolved

September7,2026,10:47 PDT. Phase2 only. Source/math work, not a new fit,
capture, estimator, numerical benchmark or score. Preserve the original
CP_FREE_NONLINEAR_TWO_MEMORY_PROPOSAL_V1.md SHA
825202e4cf4a23af275b5841e20d7120dec20e3ffb803f2eac62f5f4e42549b6.
This additive completion changes its unexecuted covariance ownership;
do not attribute it retroactively to that proposal or the closed simple
10-parameter memory model.

## One complete approximate moment-state spelling

Retain m in FP64 and L,D,H in FP32. Define the covariance state by L L.T;
its actual diagonal is the FP64 row sum of L**2. Each weight is rounded
once to W32. Promote that same rounded weight for FP64 mean/unit operations.

At an affine transition, mu=W32.T@m in FP64, B=L.T@W32 in FP32, and
Cpre=sym(FP64(B.T@B)). This uses two FP32 dense products. Form the proposal's
normalized V=diag(s_old) W32 diag(1/s), then cast its working matrix to FP32.
For E=D-(2/3)diag(diag(D)), its three-product coordinate-star action is

    Q=E@V
    Kstar=(V*V).T@Q+2*(V*Q).T@V.

The latent update U=V.T@H@V uses two further FP32 products. Keep the shared
16->8->8->2 edge MLP and the original Dpre/Hpost equations unchanged:
Dpre=Kstar+0.05*tanh(f_D) and
Hpost=tanh(U+rho*rho-diag(diag(rho*rho))+f_H).
H remains a dimensionless latent field, NOT a physically identified cumulant.

Use ZERO fourth inputs. Evaluate the Gaussian6/cubic2 covariance and raw
(2,1) channels in FP32, scaling/centering and scalar Gaussian moments in
FP64. With delta_m=k3*e/6, delta_r2=k3*d/3, M=mG+delta_m and
vE=r2G+delta_r2-M*M, require vE>0. There is no floor or retry in this
spelling. If C_B/P21 are the bank-convention connected outputs, define

    K_E = P21 - 2*M[:,None]*C_B
          - delta_r2[:,None]*delta_m[None,:]
          + 2*(M*delta_m)[:,None]*delta_m[None,:].

Restore its scalar central-third diagonal. This is literal raw-moment
centralization, including the O(k3**2) terms absent from a connected-only
formula; it is not an assertion of Edgeworth distribution positivity.

The new shared16->8->2 node head has16 inputs: alpha,g,pdf,mG/s,M/s,
vE/v,k3/s**3,diag(Dpre),diag(Hpost), Dpre rowmean/rowRMS/colmean/colRMS,
Hpost rowmean/rowRMS, and layer/15. All head operations/inputs are FP32,
clipped only at the head entrance. With outputs xi,zeta, set

    mpost=max(0,M+0.05*sqrt(vE)*tanh(xi))
    vstar=vE*exp(0.1*tanh(zeta)).

Form C_G from the positive-coefficient finite Gaussian Mehler sum with
its nonnegative diagonal tail restored. Normalize it to R_G. Form
G_H=Hpost@Hpost.T+1e-6*I, normalize to R_H, then
Rpost=(1-beta_layer)*R_G+beta_layer*R_H, with bounded learned beta in[0,1].
Scale Rpost by sqrt(vstar), symmetrize in FP64, cast to FP32 and take one
PAID FP32 Cholesky. No jitter, retry or eigensolver. Require finite output
with positive factor diagonal; that returned factor owns the next C.
The rounded input covariance is not claimed to equal its factor's Gram
bytewise or to be exactly PSD before this operation.

Retain Dpost=K_E/(vE[:,None]*sqrt(vE)[None,:]). The modeled physical plane
then uses the actual factor-owned variances, not vE. Replacing correlation
separately is an explicit approximate MOMENT-STATE splice. It is NOT
claimed to be a realizable affine transform of an Edgeworth law. First
Gaussian initialization is separate, with its exact third plane normalized
using retained-factor variances. No old P/Q, bank tensor, teacher or target
is used at inference.

## Conditional complete source census

For n1024 and16 layers, full dense FP32 products, no final-layer pruning:

| Component | FLOPs |
| --- | ---: |
| Covariance/star/latent/H-Gram skeleton,242*n**3 |259845521408|
| Edge network, inputs and outputs,518*16*n**2 |8690597888|
| Two-channel map plus separate Gaussian covariance,136*15*n**2 |2139095040|
| Casts, units, centering, mixture and guards,142*15*n**2 |2233466880|
| Separate first Gaussian initialization,180*n**2 |188743680|
| Sixteen FP32 Choleskys,16*floor(n**3/3) |5726623056|
| Node/scalar/vector work,4096*16*n |67108864|
| Additional weight ingress |33554432|
| Parameter ingress allowance |1000000|
| Conditional total |278925711248|

The142 coefficient comprises8 mean/weight conversions,6 pre-covariance,
6 V,6 rho,5 star,1 third copy,4 output casts,8 physical C/P21 scaling,
4 Gaussian scaling,12 centralization,6 third units,4 Gaussian correlation,
10 H-Gram normalization,6 mixture,10 target covariance,6 factor diagonal,
8 node matrix summaries and32 finite/domain checks. The polynomial118
coefficient is36 Gaussian+36 mixed cubic+20 local cubic+12 accumulation+
14 rho recurrence;18 more pays for a separate Gaussian covariance sum.

Root independently summed the census and read installed FNP0.12 Cholesky
and linalg precision pricing. Literal rho-squared diagonal SUBTRACTION
uses518 edge operations, not the agent's earlier517 diagonal-fill variant;
the extra16*n*n is16777216F. The literal total leaves919810160F below
the original279845521408 ceiling. FP64 dense promotion would invalidate it.

This is a SOURCE-CONDITIONED construction estimate, not a metered,
numerically qualified program or an enforceable runtime guarantee. No
actual new head/payload/candidate exists. Do not inherit the earlier
below300MiB workspace claim or infer safety from this FLOP total.

## New analytic domain witness: a later positive head cannot fix the source

Root derived this before any new data inspection; the accuracy and
efficiency agents independently confirmed it. For unit pre-variance,
alpha=-A<0, and standardized third moment t, the Gaussian raw second
moment after ReLU is

    r2G = pdf(A)*integral_0^infinity u**2*exp(-A*u-u*u/2) du
        < 2*pdf(A)/A**3.

The zero-fourth first-cubic map has r2E=r2G+t*pdf(A)/3. At A8,t=-.02,

    r2E < pdf(8)*(2/512-.02/3) < 0,
    vE = r2E-M*M <= r2E < 0.

That t is inside the .05*tanh decoder's allowed range. The positive-scale
node head occurs AFTER the vE>0 guard and Dpost normalization; moreover
an exponential multiplier cannot change the sign of vE. Thus this complete
spelling has no global safe-variance-domain guarantee.

The moment specification itself is not inconsistent: choose positive a,b
with b-a=t and ab=3-t*t, and a centered three-point law on(-a,0,b) with
probabilities1/[a(a+b)],1-1/(ab),1/[b(a+b)]. Direct algebra gives variance1,
third moment t and fourth moment3 (zero excess fourth cumulant). This is a
general moment-domain witness, NOT a claim that this discrete law is a
Gaussian-input Phase2 MLP's preactivation distribution.

No actual Phase2 occurrence, failure rate, raw error or score was measured.
The witness disproves a global numerical guarantee only, not all possible
two-memory models. Changing the positive reference scale/source law could
address this issue but would be a NEW declared model definition, not a
quiet floor or gate relaxation. No such changed model is executed here.

## Remaining two-route decision

1. CP-free nonlinear memory now has an explicit conditional cost spelling,
   potentially relevant to a below3e-9 bank if raw survives. It has NO
   demonstrated compression accuracy, the domain issue above, and no
   explicit fourth/radial feedback. The existing lean-card successes often
   retained radial state and do not establish zero-fourth equivalence.
   Resolve the state/domain law and supply a credible consumer-value
   argument before admitting a fit or coherent8-net recapture.

2. A causal joint K3/K4 correction remains the essential raw route. The
   rechecked rank35 oracle ratios(.3275 K3-offdiag,.0209 joint next-mean)
   remain privileged local evidence, not new final raw. The exact-heavy
   bridge gives.8862 and signed full recursion.8483 at roughly1.81x cost;
   no new affordable recurrence with20--50% final-raw evidence was found.
   The missing supplier is the joint nonlinear residual AFTER subtracting
   the actual bank birth/star/trace terms, with paid persistent transport.

No substantial numerical experiment is selected from these incomplete
accuracy cases. This is not an impossibility claim or external blocker.
The new result is the explicit cost/state completion and verified domain
obstruction; neither is a bank gain. Old closed families and the completed
response-row exception remain intact. Future advancement must use actual
same-configuration raw times COMPLETE scoring multiplier, not these proxies.

## Public source boundary and unchanged goal

The organizers distinguish the full fourth trace needed by their K3
construction and show a poor asymptotic ablation without the relevant
corrections: https://arxiv.org/html/2605.05179v2#A5. This does not prove a
learned H/node state cannot approximate that information; it means the
zero-fourth spelling does not inherit that guarantee. The public GitHub
search hit paulrosu11/arc-cumulant-mlp-estimator reports256x8 geometry, so
it was not treated as Phase2 performance evidence or run/downloaded.

Official#329926 remains4.982231460183304e-9; qualified independent14 local
0b84 remains4.788089831892398e-9. Immediate targetbelow3e-9; finalstrict
below1e-9, with milestone banksbelow4e-9,3e-9,2e-9. No upload, forum post,
training, MPS work, commit, promotion, frozen edit or numerical worker.
