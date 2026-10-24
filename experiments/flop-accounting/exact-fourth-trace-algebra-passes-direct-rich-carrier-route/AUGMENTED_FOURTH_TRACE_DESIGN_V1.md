# Full fourth partial-trace residual: fresh source audit before implementation

September 5, 2026. Phase 2 only. User goal remains adjusted below 1e-9.
Official usable bank remains 329926 at 4.98223146018e-9. All new learned
information/transport pilots in this continuation failed their declared
gates; no new upload is justified.

## Distinction from closed predecessors

The earlier `radial_anisotropic_fourth_source_v1.py` updated only a diagonal
of its Sym(C,D) core from row sums of the K22 plane. The earlier full-matrix
variant fitted off-diagonal K22 entries by a ridge heuristic, not by the
actual off-diagonal partial trace of the fourth tensor. Neither acquired
R_ij = sum_a kappa4(Z_i,Z_j,Z_a,Z_a), whose diagonal is only the K22 row sum.
Those variants remain closed; do not repeat their parameter sweeps.

The organizers distinguish their scalar full trace from the augmented
fourth tensor with its traceless part omitted. For order four this has
an entire matrix core. Primary source: https://arxiv.org/html/2605.05179v1,
sections4.2 andS.4.2. Local package sources were inspected read-only:
factor_k3.py / kprop_harmonic.py, mtime2026-09-04 09:26:50 PDT. Their
factorized augmentation currently materializes some (2,1,1) arrays;
the previous width1024 attempt died, so do not rerun it unmodified.

## Candidate mathematical representation (not implemented or measured)

Retain the current covariance-radial part, and add a residual Sym(A,I).
For Q=Sym(A,I),

    trace_pair(Q) = ((n+4)*A + trace(A)*I)/6,
    A = 6*R/(n+4) - 3*trace(R)*I/((n+2)*(n+4)).

Apply this only to the residual R after subtracting the existing radial
part's trace alpha*(trace(C)*C+2*C@C)/3. Under a weight push, the residual
has two matrix factors Apre=W.T@A@W and Bpre=W.T@W. It has explicit K4,
K31,K22 reads, but its K4-to-K3 source must also be accounted for; an
unadapted scalar-read injection would not be a fair candidate test.

For distinct a,i,j, the exact power-cumulant identity is

    kappa4(Y_a,Y_a,Y_i,Y_j)
      = pK211(a,i,j) - 2*m_a*pK111(a,i,j) - 2*C_ai*C_aj.

Thus trace over a may be computed before materializing a three-index
tensor. Repeated-index contributions must be restored using the same
post K31 and K22 maps. Most augmented diagrams are products of node and
pair functions on (a,i,j), so tracing a is a matrix product times a pair
factor. Third-tensor blocks can use the existing factor bank. Cancellation
between pK211 and 2*m*pK111 should be fused before expensive CP contractions.

## Immediate bounded work

1. Enumerate the exact public augmented211 diagram terms, preserving the
   distinction between full and factorized-augmentation term sets.
2. Certify a trace-first evaluator against the original dense calculation
   only at small synthetic widths (8/16), including diagonal coincidences
   and the harmonic normalization. No width1024 run before this certificate.
3. Count every matrix product and source/carry contribution. If the cost
   makes meaningful adjusted improvement implausible, stop before fitting.
4. Only after algebra/cost gates, specify a natural-coefficient two-net
   current-carrier screen with matched mean-head coadaptation if warranted.

No result, cost saving, accuracy gain, or eventual feasibility is claimed.
No forum writes, source-package edits, sealed targets, commits or promotions.
