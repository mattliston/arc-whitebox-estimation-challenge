# Tiny causal profile: prebinding-only repair is not supported

September 6, 2026, 11:30:52 PDT. Frozen program and original tiny shape
216x216 times216x432, no changed kernel or full-size benchmark. Session51292
exited0. Peak RSS73,728,000 bytes. The attempted nice10 adjustment was
refused by the sandbox; it did not apply to this 1.16-second diagnostic.
No rerun was made. Future nice adjustments require the available escalation.

Unprofiled call: context wall0.319295s = backend0.038569 + FNP
overhead0.234283 + residual0.046444. Profiled warm call0.841300s =
backend0.042690 + FNP overhead0.653550 + residual0.145060. cProfile itself
is intrusive; these are diagnostic decompositions, not new benchmark gates.

The entire kernel's Python self time is0.028557s, including arena allocate/
release/pad0.010183s. All9712 FNP getitems have cumulative0.018822s. By
contrast,3474 arithmetic-wrapper calls dominate FNP accounting/dispatch:
1564adds,1646subtracts,264multiplies, plus35matrix products and36reshapes.
Removing the arena and view bookkeeping alone cannot plausibly repair the
preserved full-size8.06x/8.62x slowdown to its3x bound. Close that specific
prebinding-only explanation; no static-view prototype follows from it.

Both calls retain original relative error1.176022e-4 and identical output
SHA9669533cfeb263e9560dd4246a4cd2a6408e2c3ad31355a5730da7a1f87a175c.
Cold50,036,224 versus warm45,963,136 FLOPs reflect pre-existing shape-view
reuse, not a new optimization. Kernel source remains
7c9ca40924410a7d5b6280c40bf500f78d9c935c831914e90047c549de430c74.
Profile source2424100394d9081c1242cd611fc3c6768101c54a5b740f73d5718c0c18834675;
profile file938b1d5e01b291fa589f57be3aa57980ee164bc7e1ccea5520f65a887bf88234.

Separate root-authorized next question: batch identical DAG operations
ACROSS recursive subproblems with a batch axis, rather than gathering
different nodes WITHIN a DAG. This changes arithmetic-wrapper count; it
still needs a complete static memory and copy-cost certificate before any
new runtime prototype. The old full-size timing failure remains unchanged.

