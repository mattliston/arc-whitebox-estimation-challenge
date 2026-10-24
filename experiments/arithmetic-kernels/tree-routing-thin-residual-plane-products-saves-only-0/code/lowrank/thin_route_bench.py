#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-TR1): in-process bench of routing the low-rank star's plain products through the alt-basis tree kernel.
Shapes (S21 census): (1024,1024,48) A@om, A^T@Q, A@Q' [45/net]; (48,1024,1024) Q^T@A [15/net]; (1024,1024,32) w^T@Q, (w^T o w^T)@B^T [28/net];
(1024,32,1024) P1@B, (.)@P1^T [28/net]. Per shape and depth L (2, 3, 4; L2/L3 regions temporarily enlarged to fit, private tree caches):
bill, op calls, user CPU per call (steady state, after a binding call), max relative difference vs the plain product; plain @ as the
reference arm. usage: thin_route_bench.py <cand.py>"""
import sys, types, time, resource, json
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
cand = Path(sys.argv[1]).resolve()
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("tr_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
rng = np.random.default_rng(0); CACHES = ("_IPT_A", "_IPT_B", "_IPT_C", "_AB_VIEWS")
def meter(c, fn, reps=3):
    out = None; f0, q0 = c._flops_used, len(c._op_log); u0 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    for _ in range(reps): out = fn()
    u1 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    return out, (c._flops_used - f0) / reps, (len(c._op_log) - q0) / reps, (u1 - u0) / reps
res = []
with flops.BudgetContext(flop_budget=2**44, quiet=True) as c:
    mod._pb_bind(time.monotonic())          # allocates the kernel arena exactly as setup does
    for m, k, n in ((1024, 1024, 48), (48, 1024, 1024), (1024, 1024, 32), (1024, 32, 1024)):
        a = fnp.asarray(rng.standard_normal((m, k)).astype(np.float32)); b = fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
        ref, fb, fq, fu = meter(c, lambda: a @ b); refn = np.asarray(ref, np.float64)
        res.append(dict(shape=[m, k, n], L=0, bill=fb, ops=fq, user_ms=1e3 * fu, maxrel=0.0)); print(res[-1], flush=True)
        for L in (2, 3, 4):
            u = 1 << L
            if m % u or k % u or n % u: continue
            region = (7 ** L * (m // u) * (k // u), 7 ** L * (k // u) * (n // u), 7 ** L * (m // u) * (n // u))
            old = mod._IPT_REGION[L]; mod._IPT_REGION[L] = tuple(max(x, y) for x, y in zip(old, region))
            saved = {x: getattr(mod, x) for x in CACHES}
            for x in CACHES: setattr(mod, x, mod._CapLRU())
            try:
                out = fnp.empty((m, n), dtype=fnp.float32)
                if not mod._ipt_ok(a, b, out, L): print("not ok", m, k, n, L); continue
                mod._ipt_product_ab(a, b, out, L)                      # binding call
                o, fb, fq, fu = meter(c, lambda: mod._ipt_product_ab(a, b, out, L))
                rel = float(np.abs(np.asarray(o, np.float64) - refn).max() / np.abs(refn).max())
                res.append(dict(shape=[m, k, n], L=L, bill=fb, ops=fq, user_ms=1e3 * fu, maxrel=rel,
                                region_changed=tuple(mod._IPT_REGION[L]) != tuple(old))); print(res[-1], flush=True)
            finally:
                mod._IPT_REGION[L] = old
                for x, v in saved.items(): setattr(mod, x, v)
json.dump(res, open(Path(__file__).resolve().parent / "thin_route_bench_S21.json", "w"))
print("DONE")
