"""TAG opus_engineer hzs certify_forced: every strip combination of the generic product and of the k-split read, FORCED on
(plan caches preset), against float64 and against the parent's bill. usage: certify_forced.py <ks fragment> <gen fragment>"""
import json
import sys
from pathlib import Path
import numpy as np
import flopscope as flops
from lab import load, Meter, rand
mod, est = load()
cases = [('gen rkc', 736, 800, 544), ('gen kc', 704, 800, 544), ('gen rc', 736, 768, 544), ('gen rk', 736, 800, 512), ('gen r', 736, 768, 512),
         ('gen k', 704, 800, 512), ('gen c', 704, 768, 544), ('push r', 736, 768, 6400), ('push rk', 736, 800, 6272), ('push k', 704, 800, 6272),
         ('push k 1pkt', 960, 928, 3200), ('ks rc', 672, 10240, 864), ('ks r', 672, 6144, 384), ('ks c', 704, 8832, 864), ('ks rc k32', 672, 5856, 864),
         ('ks k32', 704, 5856, 320), ('ks r 1chunk', 544, 2304, 384)]
res = {}
for name, m, k, n in cases:
    a = rand((m, k), 21); b = rand((k, n), 22)
    out = []
    for rep in range(2):
        with Meter() as mt:
            c = mod._mm(a, b)
        out.append((mt.flops, mt.ops))
    ref = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)
    res[name] = (out[1], float(np.linalg.norm(np.asarray(c, dtype=np.float64)-ref)/np.linalg.norm(ref)))
for f in sys.argv[1:3]:
    exec(compile(Path(f).read_text(), f, 'exec'), mod.__dict__)
mod._KS_PLANS.clear()
mod._HZ_PRICE = 0.0; mod._HZ_D5 = True
bad = 0
for name, m, k, n in cases:
    if name.startswith('ks'):
        plan = mod._ks_plan(m, k, n)
        if plan is None or len(plan) == 3:   # force a plan through the product directly
            plan = mod._hz_plan(m, k, n, True)
    else:
        mod._HZG_PLANS[(m, k, n, False)] = True
    a = rand((m, k), 21); b = rand((k, n), 22)
    out = []
    for rep in range(2):
        with Meter() as mt:
            if name.startswith('ks'):
                c = mod._ks_product(a, b, flops.numpy.empty((m, n), dtype=flops.numpy.float32), m, k, n, plan)
            else:
                c = mod._mm(a, b)
        out.append((mt.flops, mt.ops))
    ref = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)
    c64 = np.asarray(c, dtype=np.float64)
    err = float(np.linalg.norm(c64-ref)/np.linalg.norm(ref))
    worst = float(np.max(np.abs(c64-ref))/np.sqrt(np.mean(ref**2)))
    ok = err < 2e-4 and worst < 5e-3
    bad += not ok
    (pb, po), pe = res[name]
    print(json.dumps(dict(case=name, shape=[m, k, n], ok=ok, err=float('%.3e' % err), worst_abs_over_rms=float('%.3e' % worst), parent_err=float('%.3e' % pe),
                          parent_MF=round(pb/1e6, 1), new_MF=round(out[1][0]/1e6, 1), gain_pct=round(100*(1-out[1][0]/pb), 2), ops=[po, out[1][1]])), flush=True)
print(json.dumps(dict(all_ok=bad == 0, stats=dict(hz=mod._HZ_STATS, hzg=mod._HZG_STATS))), flush=True)
