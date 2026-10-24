# Signed pair-action recurrent state: static accuracy-reinvestment freeze

## Decision and boundary

`GO_STATIC_BUILD` for one target-blind source package and its exact synthetic
information gates.  `HOLD` every observer-fit target read, OOF fit, production
integration, package, board access, and submission until the source package is
independently audited and passes all gates in `PREREGISTRATION.json`.

This is an accuracy state, not an optimization claim.  It spends a small part
of packed D4's freed resource on information that the deployed 20-field tape
throws away: the *alignment* of the live post-reset K31 and K22 pair planes
with heterogeneous receiver-node shape.  No K6-completeness claim is made.

No estimator, public row, target, score, name, board, package, or network path
was opened while creating this artifact.

## Exact source law

Use production width `n=1024`.  D4 performs a full ReLU reset on layers
`l=1,...,14`; J corrects `l=2,...,15`.  Immediately after `_full_relu_reset`
on `l=2,...,14`, and before the fusion arena is reused, read without mutation:

- post covariance `C_l` with diagonal `v_l` and `tau_l=sqrt(v_l)`;
- post connected `K31_l[i,j]=cum(h_i,h_i,h_i,h_j)`;
- post connected `K22_l[i,j]=cum(h_i,h_i,h_j,h_j)`;
- post marginal `K3_l`, `K4_l`, alpha, and `tau_l`.

All sidecar arrays and learned payloads are float32.  D4's own float64/fp32
objects and statement order remain unchanged.  With

`sat(x) = x / (1 + abs(x))`, define dimensionless planes

```
rho[i,j] = C[i,j] / (tau[i] tau[j])
a[i,j]   = K31[i,j] / (tau[i]^3 tau[j])
b[i,j]   = K22[i,j] / (tau[i]^2 tau[j]^2)
r = sat(rho); q = sat(a); t = sat(b)
```

and eight ordered-pair sectors, each shape `(n,n)`:

```
E0=q, E1=q.T, E2=t, E3=r*q, E4=r*q.T, E5=r*t,
E6=q*q.T, E7=(r*r)*t.
```

K31 and K22 have exact zero diagonals in this host, so every sector is already
off-diagonal.  No identity/index sign is injected.

For every sector `s`, use two fixed, target-independent receiver probes.  The
first is one.  The second is selected from this fixed node library:

```
p_scale = sat(tau/mean(tau) - 1)
p_alpha = sat(2*alpha - 1)
p_skew  = sat(K3/tau^3)
p_kurt  = sat(K4/tau^4)
```

The second-probe map for `s=0,...,7` is
`[p_kurt,p_kurt,p_scale,p_alpha,p_alpha,p_skew,p_scale,p_kurt]`.
There are no learned pair coefficients and no fold-dependent source values.
The 16-head signed pair action is

```
A_l[i,2*s+r] = sat((n-1)^(-1/2) * sum_j E_s[i,j] * p_{s,r}[j]),
```

shape `(n,16)`, stored for layers 2--14.  `A_15` is exact zero.  The operation
is permutation equivariant: for every permutation matrix `P`, replacing
`E_s` by `P E_s P.T` and every probe by `P p` replaces `A_l` by `P A_l`.

Only `A_l`, not a pair plane, is persisted.  One target-blind D4 pass can
therefore produce a fold-shareable `(13,1024,16)` action tape per network.

## Exact learned recurrent law

Let `x_l` be the exact `(n,36)` feature tensor already formed by frozen J and
let `sigma_l` hereafter mean J's native **preactivation** standard deviation:
native raw20 plus its four-channel message/RMS, message/sigma, log-energy, and
coherence fields.  The new dimensionless state is `h_l in (-1,1)^(n,16)`,
with `h_1=0`.  Each fold owns positive training-split RMS constants `R_l`.
For `l=2,...,15`, with D4's incoming `W_l` in input-by-output convention,

```
physical_{l-1} = sigma_{l-1}[:,None] * R_{l-1} * h_{l-1}
m_l = sat((W_l.T @ physical_{l-1}) /
          maximum(sigma_l[:,None] * R_l, 2^-30))
z_l = concatenate([x_l, m_l, A_l], axis=1)              # (n,68)
u_l = tanh(z_l @ U + b)                                 # (n,64)
h_l = tanh(u_l @ V_l + c_l)                             # (n,16)
o_l = 0.25 * tanh(d_l) / sqrt(16)                       # (16,)
delta_l = sigma_l * R_l * (h_l @ o_l)                   # (n,)
reported_l = maximum(D4_plus_J_l + delta_l, 0).
```

Shapes are `U:(68,64)`, `b:(64,)`, `V:(14,64,16)`,
`c:(14,16)`, and `d:(14,16)`: exactly 19,200 learned scalars.  Parameters are
projected after every optimizer step to finite float32 with `max(abs)<=4`.
Both `sat` and `tanh` are bounded; the state has no teacher input and is
free-running at training and inference.  Initial `d=0` makes every `delta_l`
exact zero while leaving the complete source/state calculation active.

Layer order is fixed: D4 reset -> pair action capture -> D4 carrier reset and
continue; after the host finishes, frozen J is replayed in its unchanged order
while exposing copies of `x_l`, then the recurrent sidecar consumes the saved
node action tape.  This preserves frozen J arithmetic and avoids saving or
backpropagating through any full pair plane.

## Exact target-blind identifiability

### Scalar-summary collision

Let `psi=[1,-1,1,-1]` and

```
A = [[ 0,-1,-1, 1],   B = [[ 0,-1, 1,-1],
     [-1, 0, 1,-1],        [-1, 0,-1, 1],
     [-1, 1, 0,-1],        [ 1,-1, 0,-1],
     [ 1,-1,-1, 0]]        [-1, 1,-1, 0]].
```

Both are symmetric with zero diagonal.  Every row in both has sum `-1`,
squared norm `3`, absolute sum `3`, max absolute value `1`, one positive and
two negative entries.  Every rowwise and global scalar summary used by raw20,
aug7, r10, and suffix-global pooling can therefore match exactly.  Yet
`A@psi=[-1,1,-1,1]` and `B@psi=[3,-3,3,-3]`.  A signed pair contraction retains
information that no collection of those matched summaries can reconstruct.

### Physical K22 relevance witness

For a perfect matching `M` on four coordinates, draw one independent amplitude
`A_e in {1,2}` per matched edge and independent Rademacher signs `eps_i`; set
`X_i=s_i eps_i A_edge(i)` for `s=[1,2,4,8]`.  Compare
`M0={(0,1),(2,3)}` with `M1={(0,2),(1,3)}`.

Every coordinate marginal is identical between laws, covariance is diagonal,
K21 and K31 are zero, standardized marginal K4 is `-41/25`, and standardized
K22 is `9/25` on one edge per row and zero elsewhere.  Thus every marginal,
rho, K31, K22 row-sum/RMS/absolute/max/count, and global pooled summary is the
same between laws.  The scale probe is the same per coordinate but nonconstant,
so `K22_standardized @ p_scale` reveals the matching.

For the common next affine coefficients `[1,1,3/4,1/2]`, exact enumeration
gives next-ReLU means `109/32` under M0 and `7/2` under M1, a separation of
`3/32`.  Pair alignment is therefore not just distinguishable; it changes the
estimand while every deployed scalar summary matches.

## Strict deduplication

- C8/C8v2 carry 8 per-unit channels sourced only from local raw20/nonlinear-Z
  features and exact-W messages.  C8G adds pooled suffix context.  Neither
  reads one coordinate of K31 or K22.  This state has 16 channels only because
  16 fixed pair actions exist; its zero-pair matched control is mandatory.
- Q/aug7 and R/r10 export per-unit discarded-mass, row RMS, rho-power, max, and
  pooled summaries.  They never preserve which receiver node owns a K31/K22
  edge.  The A/B collision above holds all such summaries fixed.
- G11A is an off-diagonal K21/action-source spelling.  The frozen bank uses no
  K21 sector; it starts at K31/K22 and nonlinear rho-coupled versions.
- P is a memoryless `W/W.T` vector round trip of a local raw20 score.  It has no
  pair-moment input and no new recurrent state.
- The lambda3/S0 sidecar asks whether one CP-derived full plane is genuinely
  dense.  It failed its source-rank gate.  This bank consumes D4's already-live
  connected pair planes, collapses them immediately to 16 actions, and makes
  no lambda3-density claim.
- K33/K42/K51 are fixed rank-201 sixth-order pair-inheritance providers.  This
  is a learned residual state sourced by fourth-order pair topology; it neither
  retries their source law nor changes their terminal receipts.
- P2R0F3A1 is a rank-4 *physical* K22 residual carrier injected back into the
  host propagation, and its exported tape was reconstructable by J.  This bank
  does not perturb host moments: it contracts the already-live post-reset K22
  plane against fixed heterogeneous receiver probes, retains 16 observer
  coordinates, and learns only a sidecar residual state.
- P2R0F3A6B4 transports dropped R31/K4 mass as another physical carrier; its
  receipt records worse raw error.  P2R0F3A6B5 is a deterministic analytic
  order-<=4 joint-mass supply.  The present law retries neither correction: its
  target-blind source is the signed *edge ownership* of K31/K22, which their
  carrier/tape laws collapse, and the mandatory pair-zero control isolates
  that information from generic recurrent capacity.
- G17 is the support-3+ connected K6 quotient.  Pair actions cannot identify
  the P+/P- K6 witness and do not unblock G17.  This route is orthogonal: it
  tests whether alignment already present in D4's K31/K22 planes was discarded
  only by the learned observer interface.

## Frozen official resource envelope

The packed-D4 static authority keeps the hot maximum at
`1,356,595,288,584` FLOPs, projected peak `5,411,373,056` bytes, and measured
maximum residual `0.35151344118639827` seconds.  The source/state spelling is
frozen to these conservative exact caps:

- derived FlopScope-0.12 upper: `1,517,332,480` added FLOPs;
- sealed implementation cap including `2,667,520` reserve:
  **`1,520,000,000` added FLOPs**;
- at most **800** added metered array calls;
- explicit simultaneously-retained allocation upper: **62,627,840 bytes**;
- incremental measured residual must be `<0.035` seconds.

Combined maximum is `1,358,115,288,584` FLOPs, utilization
`0.6175993296819797`, with `840,907,966,968` FLOPs headroom.  Projected peak is
`5,474,000,896` bytes, leaving `3,115,933,696` bytes below 8 GiB.  At that cap,
raw must be below `2.8879912800984186e-8` to improve D4's adjusted bank and
below `1.2425667579050766e-8` to beat the live leader.  The OOF promotion gate
has two frozen outcomes: any all-fold, paired-significant adjusted win over D4
may become `PROMOTE_BANK`; only a literal-meter adjusted win over the live
leader becomes `LEADER_CANDIDATE`.  A real bank improvement is not discarded
merely because it closes less than 60% of a diagnostic q-gap.

The byte and FLOP values are exact static allocation/operation caps.  RSS and
residual remain empirical gates: allocator overhead and dispatch cannot be
certified by arithmetic.  No resource pass is claimed before a literal
rank-35/rank-279/rank-35 A/B/A meter.

## Gate order

1. Static verifier and independent audit.
2. Exact n=4 collisions, permutation equivariance, fp32/fp64 action agreement,
   and bounded two-reset synthetic stability; no public row or target.
3. Three source-only public rows for nonzero rank/novelty, then the exact D4
   rank-35/rank-279/rank-35 zero-gain five-component parity/resource gate.
4. Only after all source gates pass, generate the target-blind all-688 action
   tapes once and seal every row/hash before any target read.
5. One fourfold OOF transaction: pair-on candidate plus its paired pair-zero
   same-capacity control, fixed final epoch, no sweep/checkpoint selection;
   apply the frozen `PROMOTE_BANK`/`LEADER_CANDIDATE` outcomes separately.
6. Full integration and any package remain separate.  Root remains the only
   submitter.
