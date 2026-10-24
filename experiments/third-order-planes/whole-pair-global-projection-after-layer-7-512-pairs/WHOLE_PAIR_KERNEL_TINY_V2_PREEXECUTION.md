# Tiny v2 qualification amendment, before all numerical execution

September 7, 2026 PDT. Both the v1 tiny script and v1 inert builder remain
preserved UNEXECUTED. Core policy/source SHA86c51be6d15f8b1c8bce463a4fb649c4974a339b5bebb20b7592669075620654 is unchanged.

Two independent reviews identified fixture/reference coverage gaps, not
projection-law changes. V2 uses the SAME random draws/seed and thresholds:

- Mask incoming star off-diagonals outside active; retain every diagonal.
  This removes unservable cold entries from the tensor/query denominators.
- Apply one fixed row-coordinate permutation to the correlated fixture,
  remapping p, star, query, newborn source/indices, and using the prescribed
  noncontiguous unsorted active order. The underlying mathematical fixture
  is unchanged.
- On the same predeclared tie/duplicate and near-cancelling boundary inputs,
  compare actual physical coefficients to an explicit original-feature
  FP64 ridge solution, with the existing1e-3 coefficient gate.

This changes no selection, cut, retained rank, ridge, precision, factor,
weight, head, tolerance or numerical failure policy. Neither v1 nor v2
has run when this amendment is written. No P2 state/target access.

Run qualify_whole_pair_kernel_tiny_v2.py once with approved nice10 launch,
all three BLAS controls1 and FNP0.12.0. Its new output is
whole_pair_kernel_tiny_v2.json. Source/notes/core seals precede/follow the
run. Builder v2 merely binds this corrected fixture certificate; it still
creates the same fixed algorithm's NEW candidate_v1 only after a pass.
No candidate or score-bearing run follows automatically from tiny PASS.
