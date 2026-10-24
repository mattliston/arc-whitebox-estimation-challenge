# Scalar CP-readout span diagnostic

Status: **target-free reconstruction from sealed predictions and persisted
score summaries**. No target row was reopened and no new score was computed.

The question is whether a cheap scalar recalibration of the existing CP3
correction could plausibly close the Phase 2 gap. For two predictions `p0` and
`p1`, every scalar interpolation or extrapolation has the form
`p(t)=p0+t*(p1-p0)`. The two already-persisted endpoint MSEs and the squared
distance between the sealed prediction arrays determine this entire quadratic
exactly, without reading the target again.

At layer 16:

- along Hermite-4 to CP1792, the best scalar is about `1.407`; its inferred
  MSE is about `3.7108e-6`, only `0.31%` below CP1792;
- along CP512 to CP1792, the best scalar is about `1.642`; its inferred MSE is
  about `3.7086e-6`, only `0.37%` below CP1792; and
- the required scientific target is below `1e-7`.

At layer 2, the Hermite-4-to-CP1792 span has an optimum MSE of
`1.9215594e-8`, about `12.36%` below CP1792's already-small
`2.1925828e-8`.  This supports keeping a learnable layer-2 readout correction,
but only behind an explicit non-regression gate.  It does not weaken the
structural zero-correction requirement at the mathematically exact first
layer.

The conclusion is narrow but useful: do not spend a fit on a global or
layerwise CP-amplitude knob as the main successor. Any material improvement
must use state-dependent information that is absent from the one-dimensional
readout span. This does not rule out CP diagnostics as inputs to a learned
observer.

Run the deterministic reconstruction with the isolated Phase 2 NumPy
environment:

```bash
pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/bin/python \
  pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/readout_span_diagnostic/verify_readout_span.py
```
