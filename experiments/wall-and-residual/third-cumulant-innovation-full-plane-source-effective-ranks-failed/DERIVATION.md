# Full-plane lambda3-innovation sidecar

Status: **PREPARATION ONLY / HOLD FOR INDEPENDENT REVIEW**.  This directory
does not authorize a source run, target access, training, integration, package,
or submission.

## Question

Can information that the deployed Q24/native-J tape currently destroys by
pooling or low-rank projection be retained as one dense, signed symmetric edge-error
state and converted causally into the diffuse mean correction that owns the
remaining endpoint error?

This is an information/action discriminator, not a claim that the lambda-cubed
host should be retried.  The literal tangent spelling

\[
e_l=J F_{Q24,l}(s_{l-1})e_{l-1}
    +(F_{\lambda^3,l}-F_{Q24,l})(s_{l-1})
\]

is rejected here: to first order it is the already-measured lambda-cubed host.
The proposed sidecar never changes a Q24 moment, covariance repair, star reset,
or carrier.  It observes the local lambda3 innovation on the frozen low-order
path, carries that innovation in a separate full plane, and reads only its
future quadratic action into the reported mean.

## Exact autonomous source

At every hidden reset `l = 2,...,15`, let the frozen base Q24 pre-reset state be

\[
s_l^-=(m_l^-, C_l^-, k_{3,l},K_{21,l},k_{4,l},K_{31,l},K_{22,l}).
\]

All of these arrays already exist in the audited Q24 closure.  For
`r in {11,21,31,22}`, define the raw pair-moment innovation

\[
\Delta M_l^r=
\sum_{a+b\in\{7,9\}}\omega_{ab}(s_l^-)G_{l,r}^{ab}(s_l^-).
\]

The eight `a+b=7` channels are exactly the `L3*L4` convolution.  The ten
`a+b=9` channels are exactly `L3^3/6`.  They are channels 17:35 in the audited
lambda3 source.  `G` is the existing Q24 Mehler/Hermite pair kernel through
`t=24`; `omega` consumes only the live autonomous `k3`, full `K21`, `k4`, full
`K31`, and full `K22`.  No teacher moment, target, network identity, public
name, score, or high-host state occurs in this definition.

For S0, standardize and symmetrize without changing relative channel energy:

\[
\begin{aligned}
Z^{11}_{ij}&=\operatorname{offdiag}\operatorname{sym}
  \frac{\Delta M^{11}_{ij}}{\sigma_i\sigma_j},\\
Z^{21}_{ij}&=\operatorname{offdiag}\frac12\left(
  \frac{\Delta M^{21}_{ij}}{\sigma_i^2\sigma_j}+
  \frac{\Delta M^{21}_{ji}}{\sigma_j^2\sigma_i}\right),\\
Z^{31}_{ij}&=\operatorname{offdiag}\frac12\left(
  \frac{\Delta M^{31}_{ij}}{\sigma_i^3\sigma_j}+
  \frac{\Delta M^{31}_{ji}}{\sigma_j^3\sigma_i}\right),\\
Z^{22}_{ij}&=\operatorname{offdiag}\frac12\left(
  \frac{\Delta M^{22}_{ij}}{\sigma_i^2\sigma_j^2}+
  \frac{\Delta M^{22}_{ji}}{\sigma_j^2\sigma_i^2}\right).
\end{aligned}
\]

Let `Z_l=[H Z_l^11 H | H Z_l^21 H | H Z_l^31 H | H Z_l^22 H]`, where
`H=I-11^T/n`; this is a 1024-by-4096 horizontal stack.  No per-plane RMS
normalization and no averaging is permitted in the rank, energy, or action
gates.  Consequently a tiny plane cannot be amplified into apparent rank and
opposite channels cannot cancel.  The learned successor, if the complete
target-blind replication gate passes, must freeze its four-plane source map
before targets; S0 does not select a channel or a layer.

The S0 runner imports the independently audited lambda3 module, replaces only
its four-pair reducer, evaluates all 35 kernels once, returns the sum of base
channels 0:17 to the host recurrence, and separately captures channels 17:35
at layers 4, 9, and 14.  Therefore the returned host follows the frozen base
path.  A separate base call must be byte-identical.  No lambda3/high-host
output is produced.

## Causal deployment law, conditional on replicated S0

The sidecar representation is pinned, including units and dtype.  `E_l` is
one fp32 signed symmetric dimensionless edge plane in post-activation
standardized coordinates.
Let `tau_l=sqrt(diag(C_l))` be the base-Q24 post-activation standard deviation
and let `sigma^-_{l+1}` be the next base preactivation standard deviation.
Define the scale-correct weight

\[
B_l=\operatorname{diag}(\tau_l)W_{l+1}
    \operatorname{diag}((\sigma^-_{l+1})^{-1}).
\]

Thus `B_l^T E_l B_l` first unstandardizes the old coordinates, performs the
raw weight congruence, and standardizes the new preactivation coordinates.
No learned scalar is asked to repair coordinatewise scale.  Let `D_l(i,j)`
denote the four dimensionless innovation channels and let `psi_theta` be one
shared 4-to-4-to-1 pointwise edge map with layer-specific zero-initialized
source amplitude:

\[
\begin{aligned}
E_1 &= 0,\\
P_{l+1} &= B_l^\top E_lB_l,\\
S_l(i,j)&=\gamma_l\,\operatorname{offdiag}\left[
 \tanh\{v_l^\top\tanh(A D_l(i,j))\}\right],\\
E_l&=\operatorname{fp32}\operatorname{clip}\left[
 (g_lg_l^\top)\odot P_l+S_l,-8,8\right],\qquad l=2,\ldots,15,\\
\delta\mu_{l+1}&=a_{l+1}\sigma^-_{l+1}\odot
  \tanh(\operatorname{diag}P_{l+1}),\\
\widehat\mu_{l+1}&=\mu_{l+1}^{\mathrm{native\text{-}J}}+\delta\mu_{l+1}.
\end{aligned}
\]

The reset factor is not an ambiguous learned scalar.  With
`alpha_l=m_l^-/sigma_l^-` and frozen Gaussian ReLU mass
`G_l=Phi(alpha_l)`, it is exactly

\[
g_l=(\sigma_l^-/\tau_l)\odot G_l.
\]

It therefore maps a standardized-pre perturbation through the factorized ReLU
tangent and expresses it in standardized-post coordinates.  `g_l`, `tau_l`,
and `sigma^-_l` are deterministic target-free base-Q24 marginals.  The first
zero conjugation is skipped; useful dense conjugations are `h2->h3` through
`h15->h16`, exactly fourteen.  The final source at h15 acts through
`P_16=B_15^T E_15 B_15`.  The scale-adjusted congruence is exact bilinear
transport of this standardized representation.  The gate sandwich is a
learned lawful closure and is deliberately **not** called the exact derivative
of Q24's nonlinear reset or PSD projection.

The Q24 host is frozen.  The identity/physical-injection setting that would
turn this back into the lambda3 host is outside the family.

Zero-gamma parity is a separate mandatory implementation invariant.  With
`E_1=0` and every `gamma_l=0`, induction gives every `E_l`, `P_l`, and
`delta_mu_l` exactly zero.  A future implementation must take a literal
zero-gamma bypass that returns the unmodified native-J output buffer, and must
prove it byte-identical before any training.  Merely adding a computed zero is
not accepted as that byte-parity proof.

## FLOP ledger

For `n=1024`, one dense matrix multiplication costs
`n^2(2n-1) = 2,146,435,072` FLOPs under the frozen ledger convention.  Two per
transition for fourteen transitions cost exactly

\[
2\cdot14\cdot 1024^2(2\cdot1024-1)=60,100,182,016.
\]

The target-free lambda3 A/B receipt measures the eighteen-channel source
increment exactly at `164,299,528,192` FLOPs.  Thus the optimistic exact core
increment is `224,399,710,208`.  The honest frozen hard cap is
`230,000,000,000`, reserving `5,600,289,792` for four-plane standardization,
the 4-to-4-to-1 pointwise map, gate products, clips, diagonal reads, output
adds, and association-preserving streaming.  A future literal meter must come
in at or below this cap; the cap is not permission to delete source channels.

Exact total bounds for call order `[35,279,35]` are:

| host | optimistic totals | honest hard-cap totals | worst budget margin |
|---|---:|---:|---:|
| D4 | `[1544271420791, 1580994998792, 1542348669440]` | `[1549871710583, 1586595288584, 1547948959232]` | `612427966968` |
| packed D5 | `[1496437376149, 1531609057800, 1492962728448]` | `[1502037665941, 1537209347592, 1498563018240]` | `661813907960` |

Every total is strictly below `2^41 = 2,199,023,255,552`.

## RSS and array ledger

The optimistic incremental live allocation is `159,383,552` bytes: eighteen
additional fp64 shell-weight planes (`150,994,944`) and two fp32 edge buffers
(`8,388,608`).  The honest hard increment is `536,870,912` bytes and is valid
only for a production reducer that never forms the audited module's
`4x35x1024x1024` fp64 kernel tensor.  The required reducer evaluates one of the
four moments at a time and at most three adjacent shell channels at a time.
Its incremental ledger is: 144 MiB shell weights, 24 MiB three-kernel block,
32 MiB four shell accumulators, 16 MiB fp32 `E/P/source/scale` buffers, at most
64 MiB normalization and association-preserving work, leaving 232 MiB of the
512 MiB cap for allocator/fragmentation reserve.

The frozen S0 diagnostic intentionally imports the audited lambda3 module and
therefore may materialize one
`4*35*1024^2*8 = 1,174,405,120` byte fp64 kernel tensor; this is its largest
diagnostic array and remains below the 4 GiB single-array cap.  Its only RSS
gate is the process-wide 8 GiB cap.  A diagnostic pass is **not** a production
RSS proof.  Any deployment source must first implement and literally audit the
streaming reducer above.  In that production ledger the largest incremental
bank is the 144 MiB shell-weight bank (and the live kernel block is 24 MiB).

Applying that bound gives D4 at most `7,270,940,672` bytes and the conservative
packed-D5 peak at most `7,657,963,520` bytes, both below 8 GiB.  The largest new
production source bank is `150,994,944` bytes, hence separately below the
official 4 GiB single-array limit.  These are static lifetime bounds; remote
execution requires a later literal RSS audit and is not licensed by this
receipt.

## Strict dedupe

This mechanism is outside every settled spelling below:

- `P2R0F2X` lambda3 host: fixed coefficient-one physical moment replacement,
  PSD repair, and reset.  The sidecar freezes the low host, carries a separate
  learned edge state, and reads a future mean action.
- `P2R0F2N`: diagonal plus rank-16 covariance factor.  The sidecar is one
  explicit rank-1024 plane and S0 removes even the best rank-16 approximation.
- `P2R0F2P`: memoryless vector `W/W^T` round trip on 36 local features.  The
  sidecar is recurrent, matrix-valued, and uses a quadratic diagonal read.
- `P2R0F2Q/R` and `P2R0G12C8G`: row/global pooled uncertainty/context.  The
  sidecar preserves signed `(i,j)` orientation.
- `P2R0G11A`: rank-4/8/16 reverse K21 action.  The sidecar uses the audited
  lambda3 pair innovation, not K21 itself, and no low-rank action.
- `P2R0G12C8/C8V2`: eight vector carriers.  No edge plane or quadratic action
  exists in C8.
- `P2R0F3A6B4/B5`: fixed physical reinjection of reset residuals/order<=4
  supply.  The sidecar changes no host state and acts directly on mean.
- Historical causal pair charts pooled/recursed pair features into vector or
  Gram summaries; none retained this signed shell innovation as a dense
  autonomous plane with `diag(W^T E W)` action.

Bound evidence hashes:

- lambda3 A/B: `ea9f7596578743bfba5c88a7d439fe7152779d46d9745171ef8b87f40ce39c17`
- lambda3 closure: `4050132ac99f04d4bcbf8c86725d55afffaa27de963b6be8d1eebfe3f58d6347`
- rank-16 covariance CV: `37187da1fabdc84951a5bc68ef3f12fba1d4b940bc19027c8426f59f1b61be37`
- full-rank vector round-trip CV: `b11009bcf3168099cf284a0614211a7821d446a0feec73e5ab222440bcd504aa`
- native R10 pilot: `40e29efcbf073e5222aa18162cc6f11a9d7a5849dc3929be628f672ae344d3fc`
- reverse K21 audit: `cd1a3f8412d1c46da8c92184bf13785b7aea19d7225dd3a0c6ce6a06ad2fb20c`
- C8v2 result: `3fa6b70c9b2f0cc799470a7443b488b3f9cfff48ad652c134c429282827f472a`
- C8G terminal result: `2c5b578182dc132f27d63378ab84f2866b85c3ffaf1c2d32f8376ff05dd33123`
- discarded-reset raw probe: `6b47f4875185350ccbaeb5d67ab5b50d8bd1d76957c0a7cecc643d01f3208ba1`
- remaining order<=4 routes: `ca559f11483c8a54e4bc5a43739eb56eef05d263ea4328271f043306c94a8d33`

## S0 decision

S0a is exactly one observer-fit rank-35 source call plus one separately freed
base-parity call.  It captures layers 4, 9, and 14 and opens no target.  For
each layer and plane let `b_l^r` be the corresponding standardized base moment
plane and define

\[
\eta_l=\sqrt{\frac{\sum_r\|H Z_l^r H\|_F^2}
                         {\sum_r\|H b_l^r H\|_F^2}},\quad
p_k=\frac{s_k(Z_l)^2}{\sum_j s_j(Z_l)^2},\quad
r_{eff}=\exp(-\sum_k p_k\log p_k),
\]

\[
e_{>16}=\frac{\sum_{k>16}s_k(Z_l)^2}{\sum_k s_k(Z_l)^2}.
\]

For the next-weight action use the same scale-correct
`B_l=diag(tau_l)W_{l+1}diag(1/sigma^-_{l+1})` and define

\[
q_l=\sqrt{\frac{\sum_r\|\operatorname{diag}(B_l^T Z_l^rB_l)\|_2^2}
                       {\sum_r\|Z_l^r\|_F^2}}.
\]

All denominators must be finite and strictly positive.  The four raw-plane
shell/base RMS ratios are recorded diagnostics, not hard gates: a future
fixed four-plane map may lawfully give a weak component zero weight without
post-hoc channel selection.  S0a instead requires every layer's joint stacked
energy ratio `eta_l>=1e-6`, minimum `r_eff>=128` with median `>=256`, and every
`e_>16>=0.50`.  All three `q_l` values must be finite and strictly positive,
and at least two of the three fixed layers must have `q_l>=0.005`.  This is an
unnormalized, energy-weighted, cancellation-free joint test and cannot pass on
one isolated layer.

A rank-35 pass returns only `GO_TARGET_BLIND_REPLICATION`, never target-bearing
authority.  The same frozen test must then pass on predeclared observer-fit
ranks 201 and 643, with no threshold or layer change.  Only a three-network
pass returns `GO_SOURCE_INFORMATION`.  The order is mechanically fixed as
`35 -> 201 -> 643`: every release authorizes exactly one rank, rank 201's
release must pin the passing rank-35 result hash, and rank 643's must pin both
passing prior result hashes.  Any miss is `CLOSE_SOURCE_COLLAPSE`; because no
later prerequisite can then validate, no retry, continuation, channel
selection, or target access follows.

After that replicated pass, a separate preregistration may authorize one
fold-pure held-172 arm.  Its fail-fast bars are autonomous raw MSE at most
`1.5e-8`, teacher-forced birth-MSE ratio at most `0.45`, and emitted-direction
oracle at most `1.3e-8`.  Only then may four-fold OOF be considered, with pooled
raw MSE at most `1.3e-8` and every fold at most `1.5e-8`.  None of that
target-bearing work is authorized here.

The `1.3e-8` bar is scientific capacity only, not board promotion.  The exact
score law is
`mean_i[MSE_i*max(0.1,F_i/2^41)]`.  At the honest maximum fixture bills, the
conservative utilization multipliers are `0.72150000441251905` for D4
(`F=1,586,595,288,584`) and `0.69904187857537181` for packed D5
(`F=1,537,209,347,592`).  Consequently the conservative raw screening ceilings
to beat the live `7.67408396767286e-9` leader are strictly below
`1.0636290950436623e-8` (D4) or `1.0978003182459445e-8` (D5).  To reach
`7.37e-9`, they are at most `1.021483015235879e-8` or
`1.0543002108857709e-8`, respectively.  These max-fixture conversions are only
safe screening bounds; final promotion must evaluate the literal per-network
`MSE_i,F_i` aggregate and strictly beat the live leader.
