# Unimplemented terminal covariance assignment removal

September 8, 2026 PDT. Phase 2 only. Root and the existing efficiency
agent inspected the actual qualified409f source. No estimator changes,
new candidate, numerical run, metered bill or submission follows.

Reference unused_materializations_v1/estimator.py SHA256
409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f.
In effective _predict_binary, line1552 evaluates

    C = C - outer(m,dm) - outer(dm,m) - outer(dm,dm)

after the learned hook has consumed the current C/features. At layer15,
the subsequent operations update m, append m.copy(), and break. The new
C is not consumed. Skipping ONLY that assignment at depth-1 preserves
the final hook inputs and all previous-layer state updates. Removing the
terminal pair-map covariance itself would NOT be equivalent: features
still consume its clipped normalized pair statistics.

Three FP64 outer products and three FP64 elementwise subtractions cost
12*n^2=12,582,912F at n1024 in the installed FNP.12 pricing. This is
.0026998554% of the qualified mean466058731667.9286F. Source inspection
establishes dead numerical dataflow, not measured all16-byte parity,
actual complete-cost delta, runtime or residual-wall benefit.

If later qualified on the same14-net panel with this constant delta and
all multipliers still above.1, the conditional adjusted value would be
4.7701069261904325e-9, using

    old_mean_adjusted - deltaF/2^41 * old_mean_NCraw.

The qualified score remains4.770235434670105e-9. This projection is not
a score gain. It is also not a multiplication of independently averaged
raw and FLOPs.

This assignment is disjoint from the still-unimplemented denominator
reuse in DENOMINATOR_REUSE_HELD_COMPONENT_2026_09_08_V1.md. Together
their source deltas sum71,332,316F (.01530543495%), implying conditional
mean465987399351.9286F and conditional adjusted4.769506922265291e-9.
Neither edit is implemented or validated; do not add either to the bank
or count the already-qualified four materialization edits again.

The earlier exact-reuse audit had identified a general dead-work envelope;
this is a precise scoped source price, not discovery of a new large route.
The DEAD component already removed outgoing post21/post22/radial work,
which is separate. Preserve this micro-component for an otherwise justified
combined implementation; no standalone run or milestone upload is selected.

Official329926 stays4.982231460183304e-9, and the official<1e-9 goal with
qualified banks below4/3/2e-9 remains active. Frozen sources and receipts
are untouched; no commit, promotion, forum post or external write.
