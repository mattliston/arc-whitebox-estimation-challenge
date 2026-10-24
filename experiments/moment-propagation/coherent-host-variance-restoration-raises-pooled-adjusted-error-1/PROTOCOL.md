# Fixed coherent restoration of pre-head active variances

Phase 2 only. Exact submitted Lazy parent, SHA256
`b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c`.
This protocol is written before numerical execution or new target access.

The learned mean hook currently preserves raw second moments, changing
centered C by -m delta^T-delta m^T-delta delta^T. This experiment retains
that mean and correlation structure but restores the active diagonal to
the physical pair-map variance immediately preceding the hook. At the END
of layers 1 through 14, after all original births and moment owners exist,
set D_i=sqrt(v_host_i/C_ii) on the actual active set; D_i=1 elsewhere.
Both active variances must be finite and strictly positive. No floor,
clipping, damping, fitted coefficient, layer selection or diagonal repair.

Apply the complete centered affine moment action:

    C <- D C D; P <- D P; b <- D b; A <- D A D^2.

A is the stored coordinate-star matrix, with T_iij=A_ji/3 off diagonal.
The shared Q is row-scaled by D; its F, core, lazy maps and coefficients
remain in their original coordinates. The pending newborn base is
row-scaled, and its two selected-coordinate offsets are scaled at their
own selected rows. No represented inherited or newborn CP owner is left
unscaled. B is derived from C and the unchanged active mask at the next
read, so it scales coherently. The radial coefficient is unchanged: its
represented fourth tensor is proportional to Sym(C,C), which scales with
four D legs. Means, lambda, masks, all learned literals, birth algorithm,
rank384, age4, lazy schedule and kernels are unchanged at this write.
Subsequent means, selectors, ranks of active boxes and bills are endogenous.

This is a fixed formal moment intervention. It does not assert the host
variance is true, repair PSD, or preserve the nonnegative support of a
literal ReLU law. FP32 factor storage retains its ordinary rounding.

Why this is distinct: the old 12-direction coherent-scale local solve used
constant/normalized skew/gate/log-sigma features and a .002 tanh envelope;
it did not use this host-to-learned variance ratio. The old current-COMP
P+C fit scaled C alone, with P trained, and failed. Centered-feedback held
the whole covariance fixed while refitting means; it also failed its gate.
Those failures remain strong adverse evidence, not unexecuted opportunities.

Stages, serial under the existing 8GB guard and zero added swap:

1. Tiny FNP tests against independent dense T/Q before and after a new
   arbitrary affine push; active/inactive coordinates, unsorted newborn
   indices, plain/lazy shared factors, null identity, domain rejection.
   FP64 C law <=1e-12 scaled error; mixed-storage tensor law <=2e-6.
2. Null source applies D=1 with identical ownership paths. Native rank35
   twice must equal ALL16 original Lazy outputs byte-for-byte and each
   other. Its added arithmetic is paid; no original-total-bill claim.
3. Sole fixed candidate: native35/100/35, then independently cold100.
   All16 finite FP32/nonnegative, unchanged first two layers, own exact
   repeats and warm/cold100 equality, original cache retirement and guards.
   Seal all outputs and complete own bills before the ordinary1B truths.
4. Admission requires candidate/parent adjusted <=.95 on BOTH35 and100,
   with their independently cold complete bills. These are reused dev
   cases, not fresh generalization. Failure closes this fixed intervention;
   no amplitude, layer, support, precision, feature, fit or source rescue.
   Passing would require a separately prepared broad panel, confirmation,
   client/runtime/contract/package checks before a preauthorized upload.

Budgets: unit60s; null150s; dev240s; cold100120s; scoring30s.
One BLAS1 nice10 numerical group; emergency guard unchanged. Artifact cap
10MB. No sampling, download, training, archive or upload in this protocol.

Source-level inverse restores the exact parent. The new helper and all
protocol/source/scorer files must be sealed before the unit stage.
