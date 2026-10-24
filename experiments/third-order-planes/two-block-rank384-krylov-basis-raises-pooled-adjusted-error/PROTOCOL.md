# Fixed rank384 two-block Krylov basis on submitted Lazy

Prospective protocol, 2026-09-19 PDT. Phase2 only; no target or numerical
result for this candidate has been opened. Parent is accepted Lazy #331599,
source SHA b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c.
The official adjusted bank is 3.8164492657795035e-9. The approximately1e-9
goal remains active; this is one bounded component experiment, not a forecast.

At joins5..14 retain the parent's entry-frozen squared-coefficient Gram
operator exactly: G X = J (weight * (J.T X)) + Qp core (Qp.T X).
Use the same first192 columns of w32[previous,:] as Omega, then

    Y1 = G Omega; Q1 = qr(Y1)
    Y2 = G Q1
    twice: Y2 = Y2 - Q1 (Q1.T Y2)
    Qn = qr(concatenate([Q1,Y2],axis=1))

The final joint QR completes an orthonormal384 basis even for a degenerate
second block; it is chosen before any test, not as a numerical rescue.
All arithmetic is the existing FP32 FNP/kernel route. No new random seed,
oversampling, threshold, fallback, precision promotion or metric change.
For full-rank blocks in exact arithmetic the span equals [G Omega,G^2 Omega].
Rank-deficient QR may complete the span with arbitrary orthonormal directions;
no exact Krylov-span equivalence is claimed in that case. In finite precision,
the transported parent core is not asserted to be exactly symmetric or PSD.

Two192-column operator applications have the same leading dense multiply
count as one384-column application. Padding, fast-multiply shape choices,
extra reorthogonalization, QR and request overhead differ and must be paid.
No cost equality or endpoint-error guarantee is inferred from the algebra.
Keep rank384, age4, physical factors, all old-core updates, Lazy joins/maps,
learned literals, kernels and all other source bytes fixed. Replace just the
Omega/Y/QR block in LAST _shared_old384_push and append inert helpers. Prove
a whole-source byte inverse and seal source before any numerical run.

This is distinct from full-width power iteration (extra full operator pass),
the failed pair384 entry metric, and the failed rank256 full signed-tensor
sketch. Do not reopen their closed sweeps. Background: Musco and Musco2015,
https://arxiv.org/abs/1504.05477, motivates retaining Krylov blocks generally;
its approximation guarantees are not claimed for this fixed FP32 estimator.

First verify the actual candidate helpers with explicit FP64 dense Grams,
without labels. Fixed synthetic cases: full-rank joining-only, joining plus
old rank384 core, rank-deficient joining, and zero operator; dimensions448
and384, with first fixture exercising a >1024-row transpose. Seed2026092005.
Require scaled relative operator error<=2e-4, Krylov-block inclusion
error<=2e-4 where meaningful, final normalized orthogonality<=5e-6, finite
FP32 outputs of shape(n,384), no input mutation, repeated arrays and complete
fresh-module bills identical. Shape/dtype mismatches must reject. Use the
actual inherited fast kernel, not a substituted dense implementation.

If certificates pass, run native35/100/35 then independent cold100. Require
finite nonnegative16x1024 outputs, first5 parent layers byte-exact, exact own
repeat and warm/cold100 parity, original cache retirement, <8GB RSS, zero
added swap. Save and seal own complete bills and predictions before scoring
the ordinary1B35/100 means against saved exact Lazy cold predictions.
Advance only if BOTH raw and adjusted scores improve on BOTH networks AND
pooled mean(raw*max(.1,own_complete_F/2^41)) <= .99 of Lazy. These reused
development networks are not independent generalization evidence.

On failure, close this fixed construction without rank, split, iteration,
seed, layer, metric, QR, precision, head, blending or fit search. On success,
predeclare a wider paired panel and actual client/contract/package checks
before any submission. Future qualified Phase2 submissions are already
authorized by the user; this protocol itself authorizes no failed upload.

One nice10 BLAS1 numerical group at a time. Existing8GB process-tree and
256MiB added-swap watchdog; qualification requires ZERO added swap. Unit90s,
three-call180s, cold10090s, score30s. New artifacts<=15MB. Preserve all old
work and root estimator. No GPU, large downloads, messages, or current upload.
