# Owner-aware correction to two unimplemented micro-component prices

September 8, 2026 PDT. Phase 2 only. This additive correction preserves
the earlier notes. No estimator, source-family experiment, numerical run,
metering test, package or submission was performed. Both proposed edits
remain UNIMPLEMENTED and UNVALIDATED.

Read this with:

- DENOMINATOR_REUSE_HELD_COMPONENT_2026_09_08_V1.md,
  SHA2af5368e307de99609b46857210ad4944795c6928aaf14779a7ccc7609f67883;
- FINAL_COV_DEAD_ASSIGNMENT_HELD_COMPONENT_2026_09_08_V1.md,
  SHAdfd56e3b446193b42af5f7bc143466161ed30ad27f01df4335a7585a1323c4cf;
- TERMINAL_SUPPORT_GRADIENT_SOURCE_AUDIT_2026_09_08_V1.md,
  SHA6f1cce61c9b64a670b7f2bf64553c84adc136dcccc84312825ef3be9840f4095,
  whose route1 micro-price repeats the former estimate. Its support algebra
  and separately labelled FP64 planning subtotal are unaffected.

The efficiency agent found two errors in the initial source prices. Root
independently inspected the installed implementations and recomputed the
conditional arithmetic. Paths below are relative to
data_gate/.venv/lib/python3.11/site-packages/flopscope in the Phase2 tree.

## Self-outer symmetry

_pointwise.py outer at5863--5952 captures original operand identity and
uses the symmetric n(n+1)/2 base accumulation count for outer(dm,dm).
The result wrapping is trusted and does not add a charged ownership copy
(_symmetric.py:702--762). The other two outers are dense. With FP64 rate2
and the three dense subtractions, the dead final C assignment costs

    11*n^2+n = 11,535,360 F,

NOT12*n^2. This is.0024750872% of the qualified mean. The conditional
unchanged-output adjusted value is4.770117624772359e-9; mean complete
cost would be466047196307.9286F.

## Trace scalar return

_counting_ops.py trace returns the underlying NumPy scalar at119.
_ndarray.py wrap_module_returns at1210--1269 wraps ndarray returns, not
that scalar. Therefore the existing expression's trace(C)**2 does not
invoke a counted fnp.power and should not have been assigned32F.
This is faithful pricing of the current source, not a proposed scalar
bypass or permission to move other arithmetic outside FNP.

The repeated denominator's actual source count is

    trace 2*n + C*C 2*n^2 + sum 2*(n^2-1)
      + final counted scalar multiply/add 4
    = 4*n^2+2*n+2.

Across14 calls this is58,748,956F, NOT58,749,404F. Conditional mean
complete cost is465999982711.9286F, adjusted4.769635435320359e-9.

## Corrected combined projection, not a new score

The two disjoint source deltas sum70,284,316F (.01508057059%). Conditional
mean cost465988447351.9286F, adjusted4.769517625422612e-9. The arithmetic
is old_mean_adjusted - deltaF/2^41 * old_mean_NCraw, valid only if the
constant complete delta and identical outputs later qualify on every net
and each multiplier stays above.1. Do not multiply separate mean raw/cost.

Current qualified409f local14 remains4.770235434670105e-9; official329926
remains4.982231460183304e-9. No gain is newly validated. No standalone
micro-qualification or milestone upload is warranted. All previous runtime,
residual-wall, RSS, fidelity, initialization and release checks remain due
for any future otherwise-justified combined implementation.
