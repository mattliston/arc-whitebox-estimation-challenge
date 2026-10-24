# Q24 CP3 fixed-rank old-summary discriminator v1

## Question

Does chronological retirement fail because it drops the wrong old CP3
directions?  At the same 2,048-column physical rank as the failed `w2` arm,
retain one exact newest 1,024-column birth and use the other 1,024 columns as
an old-state summary.

## Frozen rank-0 arms

- `prefix2`: retain the earliest 1,024 currently available old columns, then
  append the exact newest coordinate birth.
- `energy2`: retain the 1,024 old columns with largest post-gate symmetric-CP3
  self energy

  `lambda^2 * (3 ||p||^4 ||q||^2 + 6 ||p||^2 (p.q)^2)`,

  then append the exact newest coordinate birth.
- The completed age-retirement `w2` receipt is the equal-rank newest-only
  baseline and is not rerun.

At every selection boundary, all entering CP3 columns contribute to the
current K21 reset.  The dense cubic star receives
`K21_target - K21(selected old + exact birth)`, exactly paralleling the
incumbent CP4 dropped-plane residual.  The frozen native-J payload continues
to see its logical full-birth rank feature; physical rank and selected birth
ages are diagnostic-only.

## Authority and access

One already-open observer-fit network (role rank 0 / mlp id 383).  Target is
loaded only from the already-public rank-0 diagnostic artifact for evaluation.
No fitting, training, observer selection, network access, or submission.

## Fail fast

The selector is terminal at rank 0 unless native-J final MSE is at most
`8.0e-8` and improves the equal-rank `w2` value `4.166148553353033e-7` by at
least 2x.  A selector that passes rank 0 still requires a disjoint frozen
observer-fit validation panel before any deployment work.

## Cost envelope

The physical CP3 rank is exactly 2,048 after warm-up.  Selection reductions
and stable sort are receipted but are not assumed free.  The prototype may use
fresh arrays and one extra retained-subset read per reset; production cost
must be rebilled separately.  No arm may exceed `2^41` literal FLOPs or 120 s.

