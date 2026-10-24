# P2R0C1D5 independent Strassen--Winograd red-team

Status: **PASS_STATIC; AUTHORIZE_TARGET_FREE_IMPLEMENTATION_AND_METER_ONLY**  
Date: 2026-08-26  
Scope: exact `#328439` estimator, CP3 contractions only; no targets, production
forward, graph edit, network action, or submission were used by this audit.

## Verdict

The proposed route has no static blocker.  I independently reproduce the
58-call inventory, literal FlopScope arithmetic, memory-scheduled depth choice,
and legality classification.  It is worth spending the next experiment on a
sealed target-free implementation and common-driver meter.

This is not a meter pass or an accuracy claim.  The 7.3601-GiB figure is an
algebraic live-array bound with only 687,046,656 bytes of margin to 8 GiB.  It
does not certify allocator high-water behavior.  Residual wall and the effect
of depth-4 float32 reassociation through the full recurrence are likewise
empirical gates.  No target score or submission should be authorized from the
static evidence.

## Independent findings

- The bound source SHA-256 is
  `373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7`.
  Its CP3 loop has 30 push products and 28 nonfinal read products.  The final
  marginal `_cp_k3` matrix-vector read is correctly excluded.
- The ordinary CP3 bill is exactly `966,086,623,232`: push
  `515,144,417,280`, read `450,942,205,952`.
- The 15-add Winograd schedule, including every add/subtract, seven-way stack,
  block/top-output materialization, and leaf matmul, bills
  `594,693,332,992` at depth 4.  Ten common-left paired-push stacks add
  `20,971,520`.  The resulting net saving is exactly
  `371,372,318,720`.
- The rank-gated call schedule has 20 push kernels and 17 read kernels.  A
  normal depth-4 kernel has 73 operations.  A push replaces the top block with
  two direct `concatenate(..., out=destination_slice)` calls, so it has 74;
  ten paired pushes add one common-left stack each.  Therefore the exact
  kernel-related inventory is push `1,490` plus read `1,241` = `2,731`.
- The installed meter accepts `concatenate(..., out=...)` and charges exactly
  the destination element count.  The proposed schedule therefore does not
  need an uncounted copy or TD52-style operator-price route.
- A separate paired depth-4 synthetic push used the direct-output spelling.
  It billed exactly `431,896`, matching the independent formula, with exactly
  75 operations (`32 subtract`, `28 add`, `9 stack`, `3 block`,
  `2 concatenate`, `1 matmul`).  It was finite; maximum relative error versus
  float64 matmul was about `3.75e-6`.  This validates arithmetic and billing,
  not production drift.
- The memory formulas reproduce 6,798,163,968 bytes for the largest sequential
  depth-4 push, 7,902,887,936 for the largest paired depth-4 push,
  7,838,334,976 for the largest paired depth-4 read, and 9,353,728,000 for the
  largest sequential depth-5 push.  Depth 5 is closed for this frozen
  vectorized schedule, while depth 4 still requires literal RSS measurement.
- This is genuine scalar-operation reduction under ordinary float32
  `matmul/add/subtract/stack/block/concatenate` prices.  It is in the same
  clean legality class as classical Strassen, and does not use tensordot,
  singleton-axis pricing, meter mutation, off-meter arithmetic, rank
  truncation, or hidden state changes.

## Required execution protocol

Run one fresh process with one persistent estimator instance and the frozen
call order `[35,279,35]`.  "Cold" must not be interpreted as three independent
processes: persistent scratch reuse and repeat-rank byte parity are part of the
governing common-driver protocol.

Every call must independently pass the preregistered bill, shape, finiteness,
RSS, outer-wall, and residual-wall gates.  The repeated rank-35 candidate must
be byte-identical.  Candidate/incumbent comparisons must use the same
prebound weights-only fixtures and report all per-layer RMS/max deltas.  Any
extra materialization in the implementation must appear in the literal bill;
the exact `371.372318720B` saving assumes only the frozen common-left pack on
top of the fully counted kernel.

If and only if all target-free gates pass, the route becomes eligible for a
separately authorized target score.  At unchanged accuracy its aggregate-ratio
planning projection is still about `1.788e-8`, so it does not itself solve the
sub-`1e-8` objective.

## Reproduction

```bash
python3 pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/independent_redteam_check.py
pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/bin/python pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/independent_redteam_check.py --micro
```

Expected statuses are `PASS_INDEPENDENT_STATIC` and
`PASS_INDEPENDENT_TARGET_FREE_D4_MICRO`.
