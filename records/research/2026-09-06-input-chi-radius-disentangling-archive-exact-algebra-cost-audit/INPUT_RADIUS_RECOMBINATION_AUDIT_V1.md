# Input chi-radius disentangling: archive, exact algebra, and cost audit

September 6, 2026. Phase 2 only. No new targets, fit, network replay,
sampling, MPS work, radius schedule or inference implementation.

## The actual spherical experiment already ran

File mtimes, not handoff dates:

- radial_compact_sphere_v1.py: September4,20:46:35PDT.
- chain_compact_sphere_training_v1.json: September4,21:11:00PDT.
- compact_sphere_closedloop_r35_v1.json and r100: September4,21:14:12PDT.

The code explicitly propagates radius-sqrt(n) UNIFORM-SPHERE input,
initial C=I, zero third cumulant, radial fourth coefficient -6/(n+2).
It multiplies all output means by E[chi_n]/sqrt(n). It is not a Gamma
hidden-variance consumer and uses no cubature points.

The honest row generator captures angular states before correction,
converts saved host/sigma to Gaussian units, and divides Gaussian mean
feedback by the known radius mean. All200 rows and the fixed160/40,
C64,40-epoch matched fit completed; total1002.19seconds. The full-final-
feature closed-loop evaluations were:

| Development net | Angular raw final MSE | Matched Gaussian parent | Ratio |
| --- | ---: | ---: | ---: |
| rank35 | 2.0281909707144275e-8 | 2.0444447453558874e-8 | .9920497853129161 |
| rank100 | 1.9869130913426062e-8 | 2.0040627884869033e-8 | .9914425350129646 |

These existing receipts used separate1e9 mean references, not the14-net
grader panel. No panel, metered port, radius/epoch sweep or follow-up is
licensed by the ~.8% gains. The initial angular Edgeworth map was also
explicitly known to be approximate: its first-layer Gaussian-frame mean
error was2.016852e-7, matching its derived O(n^-2) truncation term.

## Exact recombination through fourth order

For every bias-free ReLU prefix H, positive homogeneity gives

```
X = R U,    R ~ chi_n,    U uniform on the unit sphere, independent
A = H(sqrt(n) U),    S = R/sqrt(n),    H(X) = S A.
```

For any raw order-k tensor, M_k^G = a_k M_k^A, exactly, where

```
a_k = (2/n)^(k/2) Gamma((n+k)/2) / Gamma(n/2)
c = a_1,    a_2=1,    a_3=c(1+1/n),    a_4=1+2/n.
```

Let mu,C,T,Q be the ANGULAR mean/covariance/third/fourth cumulants and
d=1-c^2. Define Sym3(mu,C) as the three distinct placements, Sym3(C,C)
as the three pairings, Sym4(mu,T) as four placements, and Sym6(mu,mu,C)
as six placements. Then the Gaussian-input cumulants are exactly

```
mG = c mu
CG = C + d mu^2
TG = a3 T + (c/n) Sym3(mu,C) + beta mu^3
QG = a4 Q + (2/n) Sym3(C,C)
     + (a4-c*a3) Sym4(mu,T)
     + (2*d/n) Sym6(mu,mu,C) + kappa4S mu^4

beta = a3 - 3*c + 2*c^3 = c*(1/n-2*d)
kappa4S = a4 - 4*c*a3 - 3 + 12*c^2 - 6*c^4
        = 4*d-2/n+4*d/n-6*d^2.
```

Powers of mu denote tensor products. The full fourth-order identity
contains Sym(mu,T), not merely a radial covariance-pairing tensor. It
must not be omitted when claiming exact central-moment recombination.
Inversion is triangular: divide RAW tensors by a_k and re-center.

A tiny finite-atomic A distribution and analytic chi moments provide an
independent partition-cumulant certificate through k=4. This checks the
newly written central identities only; it is not a network experiment.

Completed certificate: input_radius_recombination_certificate_v1.json.
All raw-order/cumulant and inverse checks pass, maximum absolute
disagreement9.64e-16,0.0099seconds,42,696,704bytes. The two input dimensions
16/1024 occur only in an algebra identity on a fixed three-dimensional
atomic output law; no Phase1 network was loaded or evaluated. At n=1024,
c=.9997558892136928, d=.0004881619825383952,
beta=2.384766942619451e-7,
a4-c*a3=.0014652012032244678, kappa4S=1.7069648435085733e-13.
The old gammaln-based c differs by3.03e-13, not a meaningful score lever.

## Is this redundant with the current bank?

Not as a literal state identity. The Gaussian-input bank initializes
radial=0 and evolves an effective hidden-law radial scalar and trace b;
it does not explicitly retain an independent chi_n input radius. The
third radius contribution has a trace term plus a rank-one cubic term,
which its representation can express. The fourth radius contribution
also contains Sym(mu,T), absent from a scalar-only radial fourth ansatz.
It would be incorrect to claim the bank already tracks all these exact
terms separately or that every angular closure is algebraically equal.

But exact reparameterization alone supplies NO new information or mean
gain. If D converts angular moments to Gaussian moments, a transformed
closure D^-1 o F o D with the full required state is conjugate to F.
Starting from the equivalent input and finally applying D reproduces
the same approximation, up to arithmetic. To alter accuracy one must
change the closure, discard/restore different structure, or supply
additional angular higher cumulants. The executed sphere experiment
already made the basic distinct closure choice: run the existing
truncated map directly on angular state and rescale the mean.

Exact Gaussian recombination used only at the final scored mean is just
c*mu: the executed implementation already does this. Recombining
orders2..4 at the final read cannot improve that mean by an identity.

## Cost and bounded novelty verdict

- Raw moment recombination is elementwise scalar scaling once those
  moments exist. Means O(n), covariance and existing two-index cumulant
  planes O(n^2). It creates no new inference information.
- Third recombination can scale the existing CP weights and add a
  trace term plus one rank-one cube. No full n^3 tensor is required.
- Existing-frame fourth diagonal/K31/K22 recombination is O(n^2), using
  mu,C and the corresponding third planes. Full fourth transport is
  not free: its Sym(mu,T) component must be represented/queryable across
  subsequent maps, and the nonlinear angular closure must specify the
  transformed fourth state. Omitting that ownership is not exactness.
- Starting the already executed sphere card and simply changing its
  radius factor, epochs or layer schedule is not a new mechanism.

Verdict: do NOT launch an angular rerun or a second radius fit. The
specific economical input-radius separation has already been executed
with only~.8% two-net raw gains. Exact through-fourth recombination is
useful bookkeeping, but no additional deployable gain follows from
the identity. A new coupled angular fourth closure would be a different
research question, requiring its own information/value and complete-
cost gate; this audit does not establish its value or authorize it.
