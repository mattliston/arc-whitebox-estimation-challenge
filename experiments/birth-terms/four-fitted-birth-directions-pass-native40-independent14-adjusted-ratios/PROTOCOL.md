# Fresh-eye direct birth-direction experiment

September 8, 2026 PDT; Phase 2, width1024/depth16 only. Old work and all
negative receipts remain untouched. This is a new functional-state fit,
not a claim that old negative measurements were wrong. No forum posts.

## Two routes and selection, before substantial numerical work

1. **Directly learn the newborn common-base direction. Selected.** The
   previous complete-carrier P/Q fit changed mean heads, not this state
   action. Fixed positive-path/heavy-leg sources and conditional lambda
   fits did run and were unsuccessful; they are not rerun here. Existing
   truth-supplied joint-third tests establish a large local opportunity
   (roughly70% next-mean-response error removal), not a final-score gain.
   Carried error dominates the measured third-error innovation split.
   The new intervention adjusts how that state is created, with the same
   factor count and current all16 response heads. A working hypothesis is
   a10--25% raw improvement, with HIGH uncertainty and no demonstrated
   gain. About17% at unchanged complete cost would cross the first local
   milestone. This is not a forecast of that crossing.

   The extra constructor/restoration is O(15*n^2), not another tensor
   carrier. A conservative source-operation allowance is6B over the
   current466.059B mean bill at unchanged geometry, approximately472B
   complete; changed endogenous masks can change this estimate. Allow
   up to500B in planning, not as a substitute for measuring each complete
   bill. Initialization, all P heads, restoration and conversions are
   included in the actual native judgment. Missing assumption: a cheap
   directional action actually predicts signed final error across nets.
   Cheapest decisive test: one fixed end-to-end fit of60 coefficients,
   frozen P, followed by final-output and own-complete-cost comparison.
   No isolated tensor-error threshold will substitute for that objective.

2. **Best-cube pair recompression. Not selected now.** Prior first-jet
   rejection was a tensor-proxy gate, not an unconditional final-score
   test. The old all-eligible envelope suggested about31% cost savings,
   but was old P0/Q geometry and excluded current implementation terms.
   It does not establish a current complete bill below300B. At a
   hypothetical320B, sub4 requires raw<2.749e-8 and sub3<2.062e-8.
   Current cost and final raw retention are both unknown. The decisive
   test would be a fully specified current-frame compressor rollout with
   restoration and its own bill. This has weaker evidence than the
   selected near-constant-cost raw route; no compression run is forced.

## State action and ownership

Parent is unused_materializations_v1/estimator.py,
SHA409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f.
P is payload_response_rows_v1.npz,
SHA6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c.
P remains frozen. No Q, new features, new factors, covariance decoder,
teacher forcing, or current-target-dependent input is introduced.

Let B=offdiag(g[:,None]*Cpre), d=pdf/sigma, q=alpha/sigma,
t3=diag(K21)/sigma^3. The four typed candidate directions are

    F1_ij = .5*d_i*K21_ij
    F2_ij = -.5*g_i*q_j*K21_ji
    F3_ij = -.5*d_i*q_j*Cpre_ij^2
    F4_ij = Cpre_ij*t3_i*(alpha_i^2-1)*pdf_i/6.

Zero every direction's diagonal. Independently residualize its column
projection onto B, with denominator max(sum_i B_ij^2,1e-20). Orthogonality
is exact only above that denominator floor. This removes
the scalar pair-amplitude confound, not every possible equivalent action.
DeltaB=sum_j tanh(theta[layer,j])*Fj, with60 zero-initialized parameters
for layers0..14 (60 allocated; the four layer0 directions are identically
zero under the Gaussian initialization). This does not change inherited g. F4 is only a newborn
gate-response direction, not a complete corrected cumulant algorithm.

Keep the original born plane and original t/s/a construction. Define
old/new common bases by the same masked FLOAT32 storage convention as
the actual newborn tuple. In float64 use their stored difference D and
the stored float32 a,t,s promoted to float64. Put w=a*(t-s), z=a*(t^2-s^2).
Add to the ORIGINAL born plane

    delta21_ij = w_j*births_j*(2*oldB_ij*D_ij+D_ij^2)
                 + z_i*births_i*D_ji.

Its diagonal is zero. Existing live masking/restoration then runs as
before. The newborn tuple uses newB; factors and weights are not dropped.
Using beta in place of z would silently change clipped-cell ownership.
At zero theta the stored D and increment must be exactly zero.

## Fixed fit and judgment

Use the existing ranks35..234,160 TRAIN/40 VALID split from
joint_prefix_final_data_v1.py. These are reused public development nets,
not pristine holdouts. Training means are the existing1e9 truthpost
mc_post arrays, not noisy1M variance labels. No new samples are acquired.

Six epochs, batch4 nets with immediate per-net backward and one resident
graph. Seed9309081, Adam(lr.025 cosine to.0025, betas.9/.999,eps1e-8),
no weight decay; gradient norm cap5. Optimize final MSE plus.05 times
mean layers0..14 MSE, divided by2e-8. Only theta is trainable. No
validation checkpoint selection, fit extension, coefficient sweep or
head refit in this round. Preserve audit checkpoints; judge terminal only.

Before fit: check small stored-pair algebra/null/restoration, independent
NumPy/Torch basis agreement, current all-P zero-action native/NumPy/MPS
forward compatibility on35/100, and a nonzero-action gradient/finite
response fixture. Reuse established centered MPS arithmetic; do not
mistake ordinary uncentered FP32 cumulants for a faithful derivative.
Require full-size MPS/native max output difference<=1e-5, paired MSE
ratios within1%, and a non-negligible fixed-direction parameter-gradient
relative discrepancy<=5% between CPU64 and MPS32. Numerical failures
permit versioned implementation repair, not accuracy-driven tuning.

Seal terminal coefficients and all40 validation predictions before the
separate judge reads those validation truths. Build terminal and null
native implementations and measure raw and COMPLETE F in the same
configuration. Mean per-net raw*max(.1,F/2^41) is the decision statistic.
Compare the terminal to its own null and the current bank, not a mean
raw multiplied by a mean cost. Advance to public8+virgin6 if the complete
objective improves and at least24/40 nets improve. Retain small fully
validated components; submit only milestone-qualified banks below4e-9,
3e-9,2e-9, with eventual strict board goal<1e-9. Runtime/residual/memory,
FlopScope-only contract, packaging and provenance checks remain required.

## Resources

One nice10 MPS process, at most3 Torch CPU threads, fallback disabled,
never AP+threadpoolctl. One weight graph resident, no other numeric job
during wide MPS work. The prior full fit peaked9.74GB MPS driver and
5.10GB process RSS; the48GiB machine currently has776.75MiB swap used.
Abort if driver>16GB or process RSS>12GB, or on nonfinite state/gradients;
monitor swap and memory pressure. Native qualification separately retains
the actual8GB process and120s/.4s wall/residual limits. No frozen edits,
commits, promotions or uploads above the requested milestones.
