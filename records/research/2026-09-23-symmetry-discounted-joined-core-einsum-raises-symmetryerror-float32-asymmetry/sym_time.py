#!/usr/bin/env python3
"""TAG fable-engineer (E-41): in-process time (min of 3 warm calls, server's numpy/OpenBLAS, run with 1 BLAS thread) and bill of the
symmetric alternatives priced by sym_price.py, on the candidate's own kernel. usage: sym_time.py <candidate.py>"""
import sys, time, types, warnings, runpy
from pathlib import Path
warnings.filterwarnings("ignore")
sys.argv = [str(Path(__file__).with_name("sym_price.py")), sys.argv[1]]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(sys.argv[0], run_name="sym_price_timing")
mod, fnp, X, w, C, ctx, sym_gram, sym_upper = (g[k] for k in ("mod", "fnp", "X", "w", "C", "ctx", "sym_gram", "sym_upper"))
import numpy as np
rng = np.random.default_rng(1)
F = fnp.asarray((rng.standard_normal((384, 6144)) / 32).astype(np.float32)); gv = fnp.asarray(rng.random(6144).astype(np.float32))
P = mod._mm(C, w)
n = 1024; dense = 2 * n ** 3
def run(label, f, d=dense):
    f(); ts = []; bs = []
    for _ in range(3):
        b0 = ctx._flops_used; t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0); bs.append(ctx._flops_used - b0)
    print(f"  {label:34s} {min(ts)*1e3:7.1f} ms   bill {bs[-1]/d:6.3f} units", flush=True)
print("GRAM X^T X 1024^3")
run("current _mm(X.T, X)", lambda: mod._mm(X.T, X))
run("plain matmul X.T @ X", lambda: X.T @ X)
run("einsum ai,aj->ij", lambda: fnp.einsum('ai,aj->ij', X, X))
run("block recursion 1 level", lambda: sym_gram(X, 1))
print("SANDWICH second product w^T P")
run("current _mm(w.T, P)", lambda: mod._mm(w.T, P))
run("upper-block recursion 1 level", lambda: sym_upper(w.T, P, 1))
print("WEIGHTED GRAM 384 x 6144")
run("current _shared_old384_mm", lambda: mod._shared_old384_mm(F, gv[:, None] * F.T), 2 * 384 * 384 * 6144)
run("einsum ia,a,ja->ij", lambda: fnp.einsum('ia,a,ja->ij', F, gv, F), 2 * 384 * 384 * 6144)
