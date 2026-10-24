# Frozen tiny Smirnov dispatch diagnostic

September 6, 2026 PDT. Phase 2 cost research only. Root explicitly authorized
one cProfile diagnostic of the FROZEN 216x216x432 certificate, <=2GB CPU,
no kernel changes or wide run. The prior full product's 8.06x / 8.62x
runtime failure remains unchanged. No program, grouping, shape, backend,
thread-count or cache sweep is authorized by this diagnostic.

Question: does the dynamic Python arena / reference-count / per-operand
view construction in the frozen prototype account for enough time that a
separately designed static DAG compiler is worth considering? A static
compiler would have to preserve every arithmetic operation and copy, not
batch them through unpriced extra data movement or bypass FlopScope.

Use the original normalized iid seed930066, shape216x216 times216x432,
native official FNP0.12, one CPU thread and the existing fixed stream8
Product and 1GiB scratch arena. One unprofiled warm-up and one cProfile
call on identical inputs. Input generation/reference, arena creation and
profile-result processing stay outside the measured calls. All counted
product work remains inside ordinary BudgetContext. Record the official
backend/overhead/residual decomposition and per-operation timing separately
from cProfile. Profiler timings are intrusive, not a new performance gate.

Require unchanged source SHA7c9ca40924410a7d5b6280c40bf500f78d9c935c831914e90047c549de430c74,
finite output, original <=2e-4 relative numerical certificate, exact same
output bytes in both calls, peak process RSS<=2GB. Save all measurements
and function self/cumulative times. Do not add cumulative function times
as if disjoint. No targets, full-card evaluation, MPS, kernel edits,
submission, forum write or automatic follow-on prototype.

