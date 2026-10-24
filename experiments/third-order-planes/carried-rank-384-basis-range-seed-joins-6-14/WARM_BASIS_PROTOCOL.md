# Fixed carried-basis range finder

Parent is frozen Cost75, SHA5f66f94afdd2909584527a299c3989377d374289b8f5e1f379047b4176c7521c.
The original shared384 join recomputes a one-pass range from the positive
lambda-squared Gram using384 columns of the next physical weight matrix.
Every later join already has a carried, gated rank384 basis Qp. Reusing it
as the range seed may preserve dominant old directions more consistently
while the incoming cohort still contributes through J D J.T Qp.

One fixed change: retain Omega=W[previous,:384] at the FIRST join (L5),
then use Omega=Qp at joins L6..L14. Nothing changes at terminal L15, which
has no join. The full combined Gram action, reduced QR, rotations, lambda
weights, ranks, cohort ages, reconstruction, repeated planes and all heads
remain unchanged. There is no old/new blend, extra power pass, alternate
seed, fit, tensor-exactness claim, or added matrix state. Qp need not be
orthonormal after gating; the existing QR owns the new basis. The newcomer
can be underrepresented by this seed, so accuracy must be measured.

Fixed native35/100/35 and separate cold100. Require finite valid outputs,
exact repeat, exact warm/cold100, and first SIX output layers byte identical
to the parent. Source inversion must restore the complete parent bytes.
Score against existing1B development means only after sealing outputs and
full own cold bills. Advance only if BOTH adjusted scores improve and the
pooled adjusted ratio is <=.995. A negative closes this rule without a
weight-column/seed/blend/power sweep. A positive requires the unchanged14
panel and production-client qualification before a candidate bank.

Cost75 and its pending exact-archive approval remain unchanged. One numerical
process group at a time, existing nice10/BLAS1/8GB/256MiB-swap safeguards.
