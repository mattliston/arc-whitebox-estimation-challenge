"""TAG fable-lead 2026-10-19: per-network k4 FEATURE probe for the conditional-dial question. One in-process estimator (science base,
text-patched at the four k4 band sites and the four K4L-add sites) predicts each listed store rank and records, per layer 1-14,
per-unit summaries of (mu, vv, radial, k4 before the band scale, k4 after K4L): kink fraction (|z| < 0.5), mean |z|, mean z,
mean excess kurtosis k4/vv^2 and its unit sd, K4L share, vv dispersion. Rows are written as they complete (rows.jsonl).
usage: probe_k4feat_v1.py <science candidate.py> <ranks comma list> <out dir (new)>"""
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
P2 = O.parent
candidate, ranks, out = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')], Path(sys.argv[3]).resolve()
out.mkdir()
store = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest = json.loads((store / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']

src = candidate.read_text()
BAND = "                k4 = k4 * (_S19_K4BAND[0] if l <= 7 else (_S19_K4BAND[1] if l <= 11 else _S19_K4BAND[2]))\n"
K4L = "                k4 = (_e2_append_inline((_e2_value_inline := k4)) or _e2_value_inline) + _hc_k4l\n"
assert src.count(BAND) == 4 and src.count(K4L) == 4, (src.count(BAND), src.count(K4L))
src = src.replace(BAND, "                _K4FEAT_REC(l, 'band', mu=mu, vv=vv, radial=radial, k4=k4)\n" + BAND)
src = src.replace(K4L, K4L + "                _K4FEAT_REC(l, 'k4l', k4=k4)\n")

stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
REC = {}


def rec(l, stage, **kw):
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


class MLP:
    def __init__(self, weights, rank):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, rank, 'rank-%d' % rank


def summarize(d):
    f = {}
    mu, vv, k4, rad, k4t = d.get('band_mu'), d.get('band_vv'), d.get('band_k4'), d.get('band_radial'), d.get('k4l_k4')
    if mu is None or vv is None or k4 is None:
        return f
    mu, vv, k4 = mu.ravel(), vv.ravel(), k4.ravel()
    n = min(len(mu), len(vv), len(k4))
    mu, vv, k4 = mu[:n], vv[:n], k4[:n]
    s = np.sqrt(np.maximum(vv, 1e-20)); z = mu / s
    ex = k4 / np.maximum(vv * vv, 1e-30)
    f.update(n=int(n), kink05=float(np.mean(np.abs(z) < 0.5)), kink10=float(np.mean(np.abs(z) < 1.0)), abs_z=float(np.mean(np.abs(z))),
             mean_z=float(np.mean(z)), sd_z=float(np.std(z)), mean_vv=float(np.mean(vv)), cv_vv=float(np.std(vv) / max(np.mean(vv), 1e-30)),
             mean_ex=float(np.mean(ex)), sd_ex=float(np.std(ex)), mean_k4=float(np.mean(k4)), mean_abs_k4=float(np.mean(np.abs(k4))),
             radial=(float(np.mean(np.ravel(rad))) if rad is not None and np.size(rad) else None))
    if k4t is not None and k4t.size >= n:
        k4t = k4t.ravel()[:n]
        f.update(k4l_share=float(np.mean(np.abs(k4t - k4)) / max(np.mean(np.abs(k4)), 1e-30)), k4l_signed=float(np.mean(k4t - k4) / max(np.mean(np.abs(k4)), 1e-30)),
                 mean_ex_total=float(np.mean(k4t / np.maximum(vv * vv, 1e-30))))
    return f


with (out / 'rows.jsonl').open('x') as fp:
    for rank in ranks:
        REC.clear()
        weights = np.load(store / Path(manifest[rank]['path']).parent / 'weights.npy')
        t0 = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            y = np.asarray(est.predict(MLP([weights[l] for l in range(16)], rank), 2**41), dtype=np.float64)
        row = dict(rank=rank, wall_s=time.monotonic() - t0, flops=int(ctx.summary_dict()['flops_used']),
                   layers={str(l): summarize(d) for l, d in sorted(REC.items())})
        fp.write(json.dumps(row) + '\n'); fp.flush()
        print('rank %d done in %.0f s: layers %s; L10 %s' % (rank, row['wall_s'], sorted(REC), json.dumps(row['layers'].get('10', {}))[:300]), flush=True)
print('PROBE_K4FEAT_DONE')
