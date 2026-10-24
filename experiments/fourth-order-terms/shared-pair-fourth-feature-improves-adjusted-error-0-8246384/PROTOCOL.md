# Shared-pair fourth marginal: fixed residual-information experiment

Phase2 only, 2026-09-19 PDT. Parent is submitted Lazy #331599, SHA
b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c.
The official adjusted bank is3.8164492657795035e-9. The approximately1e-9
goal remains active. This is one fixed new terminal feature, not a claim
that a complete cheap fourth-cumulant closure has been constructed.

The older SHARED_PAIR_QUARTIC_SOURCE_AUDIT_20260910_V1 documents a formal
dual-order pair representation and its missing full-source consumers,
counterterms and odd descendants. We deliberately measure a different,
explicit scope: a read-only empirical feature of the existing modeled
trajectory, fitted against the remaining final mean residual. No fourth
state is fed back into that trajectory, and no Gaussian-source completeness,
radial subtraction, gauge invariance or true-cumulant fidelity is asserted.

For each actual newborn pair at postlayer0..14, compute

    epsilon = -alpha * pdf / sigma^2
    gap = float64(stored_t32) - float64(stored_s32)
    gamma = epsilon[selected]/gap where gap!=0, otherwise0

Store this one coefficient for the positive half of the cohort; the
negative half owns its opposite. Use the ACTUAL selected roster, stored
masked t/s, BD/NODE bases, rank384 projections and Lazy reconstructions.
No additional factor push, birth, rank, age or scalar-head change occurs.
The zero-gap choice is fixed before tests. At mu0 the proposed one-cubic
Gaussian source is zero although true ReLU fourth cumulants need not be;
this is explicitly an incomplete proxy, not an exact physical tensor.

At terminal prelayer15, for each actual pair of pushed FP32 factors P+/P-,
widen to FP64 and evaluate its quartic difference by the fixed identity

    gamma * (P+ - P-) * (P+ + P-) * (P+^2 + P-^2).

Sum by the original cohort order to get q4. Do this only once, at the final
layer, in bounded cohort-sized temporaries. Use existing Gaussian aux to
define r = sigma*(alpha^2-1)*pdf/24 and z = r*q4/sigma^4. All coefficient,
cast, power, sum and feature arithmetic uses FNP and is metered. Explicit
per-helper before/after counters isolate this feature bill from the
otherwise unchanged parent's complete bill. No sampler or true moment is
used in extraction. The feature uses only the network's weights and modeled
state; prior-network numeric caches must retire as in the parent.

Small certificates: actual-FNP gap/zero-gap, opposite-coefficient quartet,
paired-versus-explicit signed quartic read, dense small fourth tensor,
final derivative/shape feature, null coefficient and repeat/input ownership.
Require finite outputs, FP64 scaled differences<=1e-10 and unchanged inputs.
Then qualification native35/100/35: all16 parent outputs byte-exact,
totalF minus featureF exactly equals the saved Lazy bill for each call,
all features exact on repeated35, finite features, retired caches,
<8GB peak RSS, zero added swap. Parent wrappers must restore by identity
after every call. Seal capture code, helper, protocol and analysis before
qualification. Technical harness repairs, if needed, must be additive and
cannot change formulas, roster or scientific gates.

If qualified, capture fixed TRAIN20 ranks35,36,38,40,41,43,44,45,46,47,49,
50,51,52,53,54,55,56,57,58, reusing cold35 qualification. Every other row
is independently cold. Save only1024-vector q4/z/r/sigma, H15 and16-row
parent output, plus hashes and paid bills. Seal every capture before opening
the original1B mean references; bind to their existing manifest and exact
weight hashes. These networks have exploration/training history; this is
exploratory network cross-validation, not fresh generalization evidence.

Five folds use sorted ordinal modulo5. Control X is sigma*[1,H15] plus r
(66 channels); extension Z is just z (one channel). Including r controls
for duplicate radial-shape information. Reuse the previous verified
train-only weighted SVD residualization and ridge1e-4 fit, with RMS floor
1e-12, relative SVD cutoff1e-7 and residual-column RMS cutoff1e-6. Weights
are own-cold parent multipliers. No penalty, feature, layer, amplitude or
seed selection. Certify duplicate features cannot change control and the
two-stage fit matches the joint normal equations. Serve FP32 nonnegative
OOF means and seal them before score aggregation.

Charge parentF+1MF for the control and parentF+MEASURED featureF+2MF for the
extension. The small readout reserves are conservative information-screen
budgets, not qualified deployment bills. Pass only with pooled adjusted
error<=.95 of unchanged, <=.97 of control, and >=4/5 fold wins against both.
A negative closes this fixed feature without changing gap handling,
coefficient law, layer subset, factors, shrinkage, ridge or feature span.
A positive permits a separate native readout and new-fit-held CHECK gate,
then wider panel/client/package validation before a preauthorized upload.

One nice10 BLAS1 numerical group at a time; existing8GB RSS/256MiB added
swap watchdog, with ZERO added swap required for qualification. Unit60s,
three-call qualification180s, individual capture100s, analysis60s. No GPU,
new samples, training corpus, download or message. Artifact cap50MB.
