"""TAG fable-lead 2026-10-19: FRESH-NETWORK fit set for the conditional k4 dial. One in-process estimator (science base, text-patched at
the four k4 band sites to record per-unit (mu, vv, radial, k4) per layer); for each fresh seed (analyst/fresh loader: iid Gaussian
1024x16 nets with 2 x 32M-sample truth halves, statistically the public nets) it predicts three times: the base (features recorded),
band 2 (L8-11) k4 scale x1.05, band 3 (L12-14) x1.05 (the module global _S19_K4BAND is read inside the layer loop, so it is set between
predicts). Cross-half U = mean((p - A)(p - B)) on the final layer; f2 = 100 (U2/U0 - 1), f3 = 100 (U3/U0 - 1). Rows appended as they
complete. usage: probe_fresh_k4_v1.py <science candidate.py> <first seed> <count> <out dir (new)>"""
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O / 'analyst/fresh')); import loader
candidate, first, count, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]).resolve()
out.mkdir()
src = candidate.read_text()
BAND = "                k4 = k4 * (_S19_K4BAND[0] if l <= 7 else (_S19_K4BAND[1] if l <= 11 else _S19_K4BAND[2]))\n"
K4L = "                k4 = (_e2_append_inline((_e2_value_inline := k4)) or _e2_value_inline) + _hc_k4l\n"
assert src.count(BAND) == 4 and src.count(K4L) == 4
src = src.replace(BAND, "                _K4FEAT_REC(l, 'band', mu=mu, vv=vv, radial=radial, k4=k4)\n" + BAND)
src = src.replace(K4L, K4L + "                _K4FEAT_REC(l, 'k4l', k4=k4)\n")
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
REC = {}
ON = [True]


def rec(l, stage, **kw):
    if not ON[0]:
        return
    d = REC.setdefault(int(l), {})
    for k, v in kw.items():
        try:
            d[stage + '_' + k] = np.asarray(v, dtype=np.float64).copy()
        except Exception:
            d[stage + '_' + k] = None


mod._K4FEAT_REC = rec
exec(compile(src, str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops
BASE = tuple(mod._S19_K4BAND)
B2 = (BASE[0], BASE[1] * 1.05, BASE[2])
B3 = (BASE[0], BASE[1], BASE[2] * 1.05)


class MLP:
    def __init__(self, weights, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'fresh-%d' % seed


def summarize(d):
    f = {}
    mu, vv, k4, rad, k4t = d.get('band_mu'), d.get('band_vv'), d.get('band_k4'), d.get('band_radial'), d.get('k4l_k4')
    if mu is None or vv is None or k4 is None:
        return f
    mu, vv, k4 = mu.ravel(), vv.ravel(), k4.ravel()
    n = min(len(mu), len(vv), len(k4)); mu, vv, k4 = mu[:n], vv[:n], k4[:n]
    s = np.sqrt(np.maximum(vv, 1e-20)); z = mu / s; ex = k4 / np.maximum(vv * vv, 1e-30)
    f.update(n=int(n), kink05=float(np.mean(np.abs(z) < 0.5)), kink10=float(np.mean(np.abs(z) < 1.0)), abs_z=float(np.mean(np.abs(z))),
             mean_z=float(np.mean(z)), sd_z=float(np.std(z)), mean_vv=float(np.mean(vv)), cv_vv=float(np.std(vv) / max(np.mean(vv), 1e-30)),
             mean_ex=float(np.mean(ex)), sd_ex=float(np.std(ex)), mean_k4=float(np.mean(k4)), mean_abs_k4=float(np.mean(np.abs(k4))),
             radial=(float(np.mean(np.ravel(rad))) if rad is not None and np.size(rad) else None))
    if k4t is not None and k4t.size >= n:
        k4t = k4t.ravel()[:n]
        f.update(k4l_share=float(np.mean(np.abs(k4t - k4)) / max(np.mean(np.abs(k4)), 1e-30)), k4l_signed=float(np.mean(k4t - k4) / max(np.mean(np.abs(k4)), 1e-30)),
                 mean_ex_total=float(np.mean(k4t / np.maximum(vv * vv, 1e-30))))
    return f


def predict(w, seed):
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(MLP([w[l] for l in range(16)], seed), 2**41), dtype=np.float64)
    return y[15], int(ctx.summary_dict()['flops_used'])


with (out / 'rows.jsonl').open('x') as fp:
    for seed in range(first, first + count):
        w = loader.weights(seed); t = loader.truth(seed, check_weights=False)
        A, B = t['mc_post_A'][15].astype(np.float64), t['mc_post_B'][15].astype(np.float64)
        t0 = time.monotonic()
        REC.clear(); ON[0] = True; mod._S19_K4BAND = BASE
        p0, fl0 = predict(w, seed); feats = {str(l): summarize(d) for l, d in sorted(REC.items())}
        ON[0] = False
        mod._S19_K4BAND = B2; p2, _ = predict(w, seed)
        mod._S19_K4BAND = B3; p3, _ = predict(w, seed)
        mod._S19_K4BAND = BASE
        U = lambda p: float(np.mean((p - A) * (p - B)))
        u0, u2, u3 = U(p0), U(p2), U(p3)
        row = dict(seed=seed, U0=u0, f2=100 * (u2 / u0 - 1), f3=100 * (u3 / u0 - 1), flops=fl0, wall_s=time.monotonic() - t0, layers=feats)
        fp.write(json.dumps(row) + '\n'); fp.flush()
        print('seed %d: U0 %.4e f2 %+.3f f3 %+.3f (%.0f s) cv3 %.4f cv8 %.4f cv12 %.4f' % (seed, u0, row['f2'], row['f3'], row['wall_s'], feats['3']['cv_vv'], feats['8']['cv_vv'], feats['12']['cv_vv']), flush=True)
print('PROBE_FRESH_K4_DONE')
