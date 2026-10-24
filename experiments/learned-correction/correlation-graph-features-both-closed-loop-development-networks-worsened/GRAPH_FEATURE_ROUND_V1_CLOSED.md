# Current graph round: closed, no improvement

Completed September 4, 2026 PDT. All 200 rows passed byte-exact parent host,
sigma, and original 30-feature checks before fitting. One fixed 160/40 split,
C64, 40-epoch MPS training round; no extra rows or hyperparameter sweep.

Held-out final one-step MSE: 5.68178323279156e-9 versus frozen parent's
5.703532246492671e-9, only 0.38% improvement.
Matched full-final-feature closed loops, organizer 1e9 post means:

- Rank 35: 2.0616038857016167e-8 versus 2.0444447453558874e-8 (+0.84%).
- Rank 100: 2.065144495596603e-8 versus 2.00406278849e-8 (+3.05%).

Failed the predeclared 5% improvement on each development network.
No scalar-final refit, candidate port, 14-net panel, or submission.
Preserve the 200 rows, model, payload, and receipts; do not rerun this family.
