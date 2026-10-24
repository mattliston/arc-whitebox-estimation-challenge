# Conditional source coefficients and regression-gain scope

Phase2 only, September7,2026; completion time is the filesystem mtime.
Read-only source/math work, not a network experiment, fit or score.
Prior turn classification: progress on the scalar domain construction,
which changed the next question to missing joint information; not a score
gain. Preserve POSITIVE_MARGINAL_DOMAIN_AND_JOINT_SOURCE_AUDIT_V1.md and
every prior closeout. No old response-row exception is reopened.

## Exact next conditional coefficient

For true-centered Y, fix k with v=C_kk>0, t=T_kkk, c_i=C_ik/v,
epsilon_i=Y_i-c_i Y_k, and R_ij=C_ij-C_ik C_jk/v. Write

    P2(y)=y*y-v-(t/v)*y,
    N2=E[Y_k^4]-v*v-t*t/v > 0,
    a_i=T_ikk-c_i*t,
    U_ij=T_ijk-c_i*T_jkk-c_j*T_ikk+c_i*c_j*t,
    V_ij=Q_ijkk-c_i*Q_jkkk-c_j*Q_ikkk+c_i*c_j*Q_kkkk.

Q denotes the central fourth CUMULANT, not its raw moment. Orthogonality
of epsilon to Y_k makes the second-order conditional projections

    E[epsilon_i|y] = a_i*P2(y)/N2 + remainder,
    E[epsilon_i epsilon_j|y] = R_ij+U_ij*y/v
                  +(V_ij-(t/v)*U_ij)*P2(y)/N2+remainder.

These are orthogonal projections, not declarations that the remainders
vanish. N2=0 is a degenerate basis case, not permission to divide or floor.
The zero-variance source coordinate also needs direct handling.

The a term is available from the old repeated-third plane. Its leading
Gaussian-reference term has the already studied heavy two-edge form
e_k*(C_ik*T_jkk+C_jk*T_ikk)/2, where e_k is the scalar Gaussian third
mean derivative. Its full marginal/recentering ownership must not be
dropped. U requires full third information; V needs the residualized FULL
Q_ijkk slice. K22 and K31 do not determine the latter. Substituting the
bank's CP/radial closure computes existing model information, not the
missing true coefficient. This gives a precise new stopping point for
the conditional expansion, not a cheap Q constructor.

For symmetric source matrices M^(k), define a tensor by the unnormalized
three-placement sum sum_k SymSum(e_k,M^(k)). Its exact next repeated-plane
query for weight W is

    K'_ij=sum_k [2*W_ki*(W.T M^(k) W)_ij
                       +W_kj*(W.T M^(k) W)_ii].

Here e_k is a coordinate basis vector, not the scalar derivative above.
Rank-two hub matrices yield finite mixed factors. Generic n full matrices
require n congruences, plus their acquisition and continued propagation;
quoting one local coefficient is not a complete estimator cost. Root and
accuracy independently checked the conditional coefficients and query.

## Regression slope is not activation probability

For a true scalar law with mean mu and variance v, let m=E[ReLU(X)] and
r2=E[ReLU(X)^2]. Its linear regression slope is q_reg=(r2-mu*m)/v.
For a first-order marginal cumulant approximation, WHEN v=sigma^2,

    q_reg = Phi(alpha)+phi(alpha)*[t3*(alpha^2+2)/6
                                 -t4*alpha*(alpha^2+1)/24],
    q_gate = Phi(alpha)+phi(alpha)*[t3*(alpha^2-1)/6
                                  +t4*(-alpha^3+3*alpha)/24].

Thus q_reg-q_gate=phi(alpha)*(t3/2-alpha*t4/6). The first minimizes a
scalar linear-regression residual; the second approximates E[ReLU'(X)].
They are different objects. Approximate bank moments do not establish
orthogonality under the true law.

With truly centered F, r=F-gY, a change g->g+delta implies r->r-delta*Y.
To first order the compensating changes are

    delta E[Y_iY_j r_k] = -delta_k*T_ijk,
    delta E[Y_i r_j r_k] = -delta_j*E[Y_iY_j r_k]
                          -delta_k*E[Y_iY_k r_j],
    delta E[r_i r_j r_k] = -cyclic delta_i*E[Y_i r_j r_k].

Together with coefficient changes these cancel the inherited g^tensor3
change. Exactly, S_(g+delta)=S_g-[(g+delta)^tensor3-g^tensor3]*T.
Consequently changing the gain and ALL residual terms is a reorganization.
Changing ONLY carried CP legs with plane rebasing is a genuine estimator
intervention: its leading new term is the de-repeated cyclic
delta_i*g_j*g_k*T_CP,ijk. For q_reg it includes T_CP times marginal T,
not the previously tested T times covariance bond. It provides no new
joint statistic, and no positive current-value evidence was found.

## Do not relabel already executed transport experiments as new

Root read radial_compact_response_transport_v1.py and the scalar result
records compact_response_transport_r35_v1.json and r100_v1.json. Actual
file mtimes are September4,2026,20:37:13/14PDT. They implement q_gate
including its fourth term and clipped gate/curvature, not q_reg.

| Old unlearned full-pair arm | rank35 raw / matched parent | rank100 raw / matched parent |
| --- | ---: | ---: |
| carry |1.0113209619727754|1.004893735890449|
| carry_birth |1.021651415030686|1.0029318508879783|

Their matched baseline final MSEs are2.464131397133856e-8 and
2.6988375135721363e-8. These are archived final raw comparisons to old
1e9 references, NOT current0b84, a14-net panel or native complete scores.
The scalar-layer inherited-gain and learned-mean-response closeouts also
remain intact. No literal q_reg implementation was located in the checked
lineage, but that is not a global novelty proof or evidence to run it.

## Current implementation and cost obligations

Root inspected current0b84 active pair map and transport. The map computes
original pre-head m,r2 at lines652--653; aux708 does not expose r2. Its
gaussian_raw floors its private variance at1e-10, whereas later e/f terms
use original pre_v. The simplified q formulas above are therefore NOT
source identities when pre_v differs from sigma^2. Any candidate must
declare pre-P host, original raw2/pre_v, finite/range policy and actual
current-response semantics. It must not derive q from already corrected m/C.

Preserving newborns requires insertion AFTER their Gaussian construction.
Changing the earlier inherited plane changes residual/innovation and hence
newborns: it is a different law. The isolated construction replaces only
inherited cp_post and the existing factor-row gate. At a fixed state, an
explicit direct-r2 spelling adds6n FP64-billed scalar operations for q,
then2n+4n^2 for its additional repeated-plane calculation. Across14
nonlinear carry layers at n1024 this is58834944F, BEFORE new guards. One
extra FP64 plane is8MiB; no extra factor multiply/GEMM or column is needed.
This source increment is not a complete new-card bill: changed later
states, masks and birth counts alter the large contractions. Current
masked CP/star decomposition and rounding qualifications remain binding.

No gain-only intervention, model refit, capture, numerical probe or large
training is admitted. The missing evidence is either a compact causal
Q_ijkk/residual supplier with paid feedback, or demonstrated affordable
compression of current joint information. A mathematically different gain
alone does not fill that gap. No universal closure of all gain/state laws
is claimed, and no prospective gate has been relaxed.

## Updated future-query source boundary

The existing SCORE_AWARE_CP_PROJECTION_SOURCE_AUDIT_V1.md and
CP_SUFFIX_RANGE_PRODUCT_SOURCE_AUDIT_V1.md already cover the inspected
future-weight mechanisms. The exact endogenous trajectory is still needed
for future readers, masks and full-state risk. An extra bank pass pays
that cost; a cheaper forecast changes the approximation. No new circuit
was identified, and old lookahead/adjoint fits are not reopened.

Historical count qualification: the suffix audit's210-plane/scalar-final
wording describes old P0/Q. Current all-P0b84 uses225 shared-push and240
plane equivalents in the full-width bookkeeping. These are NOT actual
compact/D4 native bills. Root checked current all-layer hook623--634 and
shared-newborn push1287--1296; the candidate SHA remains unchanged.

## Fresh public frontier check, not new local performance

Read-only live board on September7,2026 during this turn:
https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards

| Displayed participant | Displayed adjusted | Displayed raw | Utilization |
| --- | ---: | ---: | ---: |
| Puffi |3.0e-9|2.02e-8|.1478860181|
| suliman_tadros |3.0e-9|1.97e-8|.1504215170|
| marius_binner |3.0e-9|1.76e-8|.1716246896|
| mliston, rank7 |5.0e-9|2.24e-8|.2220768611|

Scores/raw values are rounded public display values, not exact ties or
products of independently averaged columns. The page displays zero failed
MLPs for those entries; this is not a fresh per-MLP100/100 status receipt.
Leader cards did not expose method details through the public text reader.
The current forum category has no topic later than September6 in its
returned listing; no new leader implementation was located and no post made.

The freshly opened Phase2 launch notice still specifies October17,23:59UTC,
120s wall, .4s residual,8GB solution memory and the FNP-only contract:
https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197
The general challenge search extract contains superseded dates/scoring;
do not replace the existing Phase2 .1-floor protocol with that old extract.

Confirmed exact historical bank #329926 remains4.982231460183304e-9;
qualified independent14 local0b84 remains4.788089831892398e-9. No new
candidate or score gain. The frontier reinforces the immediate compute
gap without identifying the leaders' method. Final strictly<1e-9 and
milestone banks<4e-9/<3e-9/<2e-9 remain active. No uploads, forum posts,
MPS workers, commits, promotions or frozen edits occurred in this audit.
