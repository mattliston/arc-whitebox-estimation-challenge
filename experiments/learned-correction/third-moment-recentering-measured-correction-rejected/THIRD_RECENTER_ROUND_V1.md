# Bounded Phase 2 third-moment recentering test

September 5, 2026 (actual clock). Bank 329884 is pending. New target <3e-9
adjusted; no forum posts, no edits to the bank or old experiments.

The current compact host applies a learned/teacher-forced mean delta d and
keeps raw second moments fixed, C_new=C_old-m d^T-d m^T-d d^T. It leaves
the newly emitted third cumulant unchanged. This is not necessarily a bug:
the closure may be co-adapted. Test a specific alternative consistent with
keeping the raw third moment fixed as well, rather than asserting it helps.

For symmetric tensors, with Sym the SUM over distinct positions,
Delta K3 = -Sym(d,C_new)+Sym(d,m m^T)-d^3.
Represent the first term using the existing covariance-trace carrier b-d.
Represent the remainder by four balanced CP cubes, not a subtraction of
order-one cubes that would lose a small correction in float32. The current
repeated plane is updated by the exact same identity and restored by star.
No additional dense matrix multiplication is needed; four factors per layer
plus vector reductions are small relative to the existing bank. This is a
cost estimate, not a metered deployment result. Fourth moments are unchanged.

First certify the exact raw-moment identity, balanced decomposition, and
linear push on a small dense tensor. Then compare teacher-forced host error
on ranks35 and100 against the frozen same-host rows, with features read
before target feedback. One fixed full-strength variant, no scale sweep.
Require at least 10% improvement in mean one-step MSE over layers10..15
on BOTH nets to justify a matched200-row fit. If it misses, close this
variant without a fit or panel. A frozen-head loop is not a matched verdict.
