# First-join difference-aware rows, v1

Source-only prospective preparation on exact candidate 88683ee1 (Gaussian3,
plane E2, local EL2, retained precision/symmetry). No failed E1 source is included.
Execution belongs to root after review; this build does not authorize a job.

At the first old-tier join, layer7, retain exactly the first288 rows of the
incumbent stable Gram-energy ordering. Among the remaining rows choose96 by
`sum_j lambda_j^2 * (J_t[:,j] - J_s[:,j])^2`, summed over intact birth cohorts0–2.
Ties retain incumbent energy order; sort the final384 row indices as before.
All later joins use the original stable energy selection. No rank, Gram/Y,
QR, Nyström solve/core, factors, lambda, carry, consumer or young-state changes.
The only source edit is the active captured push's row-selection call, plus
one appended stateless helper. The inverse must recover the exact parent.

For p+ = u+d and p− = u−d, opposite coefficients give
lambda[(u+d)^3−(u−d)^3] = 6lambda Sym(u,u,d)+2lambda d^3,
whereas the incumbent positive Gram ranks 2lambda²(uuᵀ+ddᵀ).
The difference carries the signed cubic content even when common energy
dominates row scores. The fixed25% reserved row budget is a prospective
mechanistic hypothesis, not an optimality or endpoint-accuracy claim.

Historical evidence and limits: fable_refit_v1/TEAM_LOG.md1289 reports that
separate common/difference frames at full common384 improved old torch raw
by0.61pt on40 (26 wins), but that much larger design had different cost.
fable_covhunter_v1/RESULT.md section35 reports actual v-Gram reweighting
x3/x10/x30 losses of +.07/+.79/+1.10pt on4; section36's value-weighted L10
row selectors also failed. Those change the Gram or external row weighting;
none is this first-join-only288+96 union with the original full Gram.
No exact duplicate was found in the bounded source/receipt review. The old
young covariance-leg projection tests are negative and are not reopened.

Paid work: three cohort differences, squares and weighted matrix-vector
products; two vector additions; one extra stable sort/gather and final union.
If P is total birth pairs and m the live input rows, the leading added
arithmetic is 4mP plus O(m+P), at most12.583MF for m=1024,P=3072,
before exact vector/selection/copy charges. Runtime bills all operations.
Temporary materialization is at most two m×h float32 planes for one cohort
plus O(m+P) vectors; h≤1024, roughly8MiB of plane temporaries. No new persistent
cache, scratch arena, packed shape or setup key is introduced. This is an
accuracy hypothesis with a small paid debit, not a projected bill saving.

Root-only primitive: `NATIVE -B probe.py BUILD_SHA RESULT_JSON`, guard60s,
1GB, zeroSwap, BLAS1. Use extracted exact helper only, not estimator import.
Check all384 rows unique/sorted; exact288 protected energy rows; nontrivial
contrast selection; ties/zero contrast; later-join exact fallback; separate
cohort t-half/s-half alignment and invalid coefficients/metadata refusal;
inputs unchanged, repeated and interleaved calls exact, all paid FLOPs counted.
Reference calculations and arrays exist only inside the root-run tiny fixture.

Prospective native screen: fixed TRAIN2 ranks35,100 versus exact88683,
strict adjusted improvement and2/2 wins, all16 finite and L0–6 byte-exact.
If it passes, early boundary feasibility must pass before the fixed additional
six ranks36,45,52,58,64,70. Pooled TRAIN8 component gate: adjusted gain≥0.1%,
at least6/8 wins, paired50k-bootstrap(seed20260926) CI upper<0, and positive
adjusted gain on the additional six alone.
No raw veto. Native guards150s/300s, RSS<7.5GB, zeroSwap, BLAS1, all retired.
Final banking still needs frozen VALID40 ≥1% versus the latest official bank,
at least28 wins, CI upper<0, max per-network regression≤10%, and full serving
qualification. No row-fraction, layer, rank, score-weight or fallback rescue
after outcomes; preserve every failure.
