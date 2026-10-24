"""TAG fable-lead 2026-10-20: NATIVE Monte Carlo oracles on the shipped card's pair map (store networks 35 / 100; 32M-sample truths of
fable_covhunter_v1/mcall_rank<r>_P32M.npz: mean_pre_l, cov_pre_l, mean_post_l, cov_post_l). The live loop calls
pair_map(mu, Cpre, K21, k4, k22, metric) -> (m, C, target, radial, aux) with pair_map = nonlinear / _nonlinear_terminal_live (l = 15);
the wrapper replaces, at the chosen layers, an INPUT (cpre: the whole pre-activation covariance; vpre: its diagonal only; mupre: the
pre-activation mean; inputs: mu and C) or an OUTPUT (cpost: the post covariance; mpost: the post mean; vpost: the post variances) by
the truth, blended at 'strength' (x + strength (truth - x)). Scored on the FINAL layer against truth688 (all-layer mean secondary).
The layer index of the truth arrays is checked against the card's own diag(Cpre) (offset -1 / 0 / +1; the best is used and printed).
usage: native_cov_oracle_v1.py <candidate> <rank> <settings json: [{"layers": [..], "what": "cpre", "strength": 1.0}, ...]>"""
import json, sys, time, types
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
candidate, rank, settings = Path(sys.argv[1]).resolve(), int(sys.argv[2]), json.loads(sys.argv[3])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops
fnp = mod.fnp
W = np.load(STORE / Path(MAN[rank]['path']).parent / 'weights.npy'); W = [np.asarray(W[l]) for l in range(16)]
tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
PLN = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}.npz')      # K21_l / K31_l: diagonals = exact per-unit k3 / k4 of layer l's pre-activations
print('rank', rank, 'MC P =', int(MC['P']), 'layers with cov_pre:', sorted(int(k.split('_')[-1]) for k in MC.files if k.startswith('cov_pre_') and k.count('_') == 2), flush=True)


class MLP:
    def __init__(self, weights, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'store-%d' % seed


MODE = dict(layers=set(), what='', strength=1.0, log=[], offset=None, head=1.0); LAST = {}
def truth(kind, l):
    return np.asarray(MC['%s_%d' % (kind, l)], dtype=np.float64)
def as_like(x_np, like):
    arr = np.ascontiguousarray(x_np.astype(np.asarray(like).dtype))
    return fnp.asarray(arr) if hasattr(fnp, 'asarray') else fnp.array(arr)
def wrap(orig, force_l=None):
    def pair_oracle(mu, C, K21, k4, k22_override=None, metric='identity', **kw):
        l = force_l if force_l is not None else (int(mod._PHYS_LAYER[0]) if mod._PHYS_LAYER[0] is not None else None)
        if l is None or l not in MODE['layers'] or not MODE['what']:
            out = orig(mu, C, K21, k4, k22_override, metric, **kw)
            try:
                LAST.update(l=l, post_k3=np.asarray(out[4]['post_k3'], dtype=np.float64).copy(), post_k4=np.asarray(out[4]['post_k4'], dtype=np.float64).copy(), alpha=np.asarray(out[4]['alpha'], dtype=np.float64).copy())
            except Exception:
                pass
            return out
        Cn = np.asarray(C, dtype=np.float64); mun = np.asarray(mu, dtype=np.float64); st = MODE['strength']
        if MODE['offset'] is None:      # layer-index check: the card's diag(Cpre) against the truth's at l - 1, l, l + 1
            errs = {}
            for off in (-1, 0, 1):
                k = 'cov_pre_%d' % (l + off)
                if k in MC.files: dt = np.diag(np.asarray(MC[k])); errs[off] = float(np.linalg.norm(np.diag(Cn) - dt) / np.linalg.norm(dt))
            MODE['offset'] = min(errs, key=errs.get); MODE['log'].append('layer-index check at l=%d: rel diag err by offset %s -> offset %d' % (l, {k: round(v, 4) for k, v in errs.items()}, MODE['offset']))
        lt = l + MODE['offset']; what = MODE['what']
        mu_in, C_in, K21_in, k4_in = mu, C, K21, k4
        if what in ('k3', 'k3k4', 'allin') and ('K21_%d' % lt) in PLN.files:      # exact per-unit third cumulant: the diagonal of the K21 argument
            Kn = np.asarray(K21, dtype=np.float64).copy(); d = np.diag(Kn).copy(); dt = np.diag(np.asarray(PLN['K21_%d' % lt], dtype=np.float64)); np.fill_diagonal(Kn, d + st * (dt - d)); K21_in = as_like(Kn, K21)
        if what in ('k4', 'k3k4', 'allin') and ('K31_%d' % lt) in PLN.files:      # exact per-unit fourth cumulant: the k4 argument
            k4n = np.asarray(k4, dtype=np.float64) * np.ones(1024); dt = np.diag(np.asarray(PLN['K31_%d' % lt], dtype=np.float64)); k4_in = as_like(k4n + st * (dt - k4n), k4) if np.ndim(np.asarray(k4)) else as_like(k4n + st * (dt - k4n), np.asarray(k4n))
        if what in ('k22', 'k22x') and ('K22_%d' % lt) in PLN.files:      # the (2,2) plane: truth (k22) or the card's own x strength (k22x), through the pair map's k22_override argument
            cur = k22_override; curn = None if cur is None else np.asarray(cur, dtype=np.float64) * np.ones((1024, 1024))
            if what == 'k22':
                Tt = np.asarray(PLN['K22_%d' % lt], dtype=np.float64); base = np.zeros((1024, 1024)) if curn is None else curn; newk = base + st * (Tt - base)
            else:
                newk = None if curn is None else curn * st
            MODE['log'].append('L%d k22 in: %s -> %s' % (l, 'None' if curn is None else 'rms %.3e' % np.sqrt((curn ** 2).mean()), 'None' if newk is None else 'rms %.3e' % np.sqrt((newk ** 2).mean())))
            if newk is not None: k22_override = as_like(newk, cur if cur is not None else np.zeros((1024, 1024), np.float32))
        if what == 'allin':
            Ct = truth('cov_pre', lt); C_in = as_like(Cn + st * (Ct - Cn), C); mt = truth('mean_pre', lt); mu_in = as_like(mun + st * (mt - mun), mu)
        if what in ('cpre', 'inputs'):
            Ct = truth('cov_pre', lt); C_in = as_like(Cn + st * (Ct - Cn), C)
        if what == 'vpre':
            vt = np.diag(truth('cov_pre', lt)); Cn2 = Cn.copy(); np.fill_diagonal(Cn2, np.diag(Cn) + st * (vt - np.diag(Cn))); C_in = as_like(Cn2, C)
        if what in ('mupre', 'inputs'):
            mt = truth('mean_pre', lt); mu_in = as_like(mun + st * (mt - mun), mu)
        if what in ('k3nr', 'k4nr', 'k34nr', 'k3all', 'k34all') and LAST.get('l') == l - 1:      # own-cumulant DIAGONAL transport from the previous layer's units: k3_i += sum_a W_ai^3 post_k3_a, k4_i += sum_a W_ai^4 post_k4_a
            Wai = np.asarray(W[l], dtype=np.float64); sel = np.ones(1024, dtype=bool) if what.endswith('all') else (np.abs(LAST['alpha']) >= 1.5)      # 'nr' = only the non-roster (non-kink) units the bank has no column for
            if what in ('k3nr', 'k34nr', 'k3all', 'k34all'):
                add3 = (Wai ** 3).T @ (LAST['post_k3'] * sel); Kn = np.asarray(K21_in, dtype=np.float64).copy(); np.fill_diagonal(Kn, np.diag(Kn) + st * add3); K21_in = as_like(Kn, K21)
                MODE['log'].append('L%d k3 add rms/base %.3f (n sel %d)' % (l, np.sqrt((add3 ** 2).mean()) / np.sqrt((np.diag(np.asarray(K21, dtype=np.float64)) ** 2).mean()), int(sel.sum())))
            if what in ('k4nr', 'k34nr', 'k34all'):
                add4 = (Wai ** 4).T @ (LAST['post_k4'] * sel); k4n = np.asarray(k4_in, dtype=np.float64) * np.ones(1024); k4_in = as_like(k4n + st * add4, np.asarray(k4n))
                MODE['log'].append('L%d k4 add rms/base %.3f' % (l, np.sqrt((add4 ** 2).mean()) / np.sqrt((k4n ** 2).mean())))
        out = orig(mu_in, C_in, K21_in, k4_in, k22_override, metric, **kw)
        try:
            LAST.update(l=l, post_k3=np.asarray(out[4]['post_k3'], dtype=np.float64).copy(), post_k4=np.asarray(out[4]['post_k4'], dtype=np.float64).copy(), alpha=np.asarray(out[4]['alpha'], dtype=np.float64).copy())
        except Exception as e:
            LAST['err'] = repr(e)
        if what in ('cpost', 'mpost', 'vpost', 'outputs'):
            m, Cp, target, radial, aux = out; mn = np.asarray(m, dtype=np.float64); Cpn = np.asarray(Cp, dtype=np.float64)
            if what in ('cpost', 'outputs'): Cp = as_like(Cpn + st * (truth('cov_post', lt) - Cpn), Cp)
            if what == 'vpost': vt = np.diag(truth('cov_post', lt)); C2 = Cpn.copy(); np.fill_diagonal(C2, np.diag(Cpn) + st * (vt - np.diag(Cpn))); Cp = as_like(C2, Cp)
            if what in ('mpost', 'outputs'): m = as_like(mn + st * (truth('mean_post', lt) - mn), m)
            out = (m, Cp, target, radial, aux)
        MODE['log'].append('L%d %s applied' % (l, what))
        return out
    return pair_oracle
for name in ('nonlinear', '_nonlinear_terminal_live', '_nonlinear_first_gaussian'):
    if name in mod.__dict__: setattr(mod, name, wrap(getattr(mod, name), force_l=15 if name == '_nonlinear_terminal_live' else (0 if name == '_nonlinear_first_gaussian' else None)))
# terminal layer (l = depth - 1 with the learned head): m, aux = nonlinear_scalar(mu, v, k3, k4); m += scalar_terminal_hook(...)
if 'nonlinear_scalar' in mod.__dict__:
    _ns0 = mod.nonlinear_scalar
    def nonlinear_scalar_oracle(mu, v, k3, k4, *a, **kw):
        what = MODE['what']; st = MODE['strength']
        if 15 in MODE['layers'] and what in ('tmu', 'tv', 'tk3', 'tk4', 'tall', 'tmuv', 'tk3k4'):
            lt = 15 + (MODE['offset'] or 0)
            if what in ('tmu', 'tall', 'tmuv'): mt = truth('mean_pre', lt); mun = np.asarray(mu, dtype=np.float64); mu = as_like(mun + st * (mt - mun), mu)
            if what in ('tv', 'tall', 'tmuv'): vt = np.diag(truth('cov_pre', lt)); vn = np.asarray(v, dtype=np.float64); v = as_like(vn + st * (vt - vn), v)
            if what in ('tk3', 'tall', 'tk3k4'): k3t = np.diag(np.asarray(PLN['K21_%d' % lt], dtype=np.float64)); k3n = np.asarray(k3, dtype=np.float64); k3 = as_like(k3n + st * (k3t - k3n), k3)
            if what in ('tk4', 'tall', 'tk3k4'): k4t = np.diag(np.asarray(PLN['K31_%d' % lt], dtype=np.float64)); k4n = np.asarray(k4, dtype=np.float64) * np.ones(1024); k4 = as_like(k4n + st * (k4t - k4n), np.asarray(k4n))
            MODE['log'].append('terminal %s applied' % what)
        return _ns0(mu, v, k3, k4, *a, **kw)
    mod.nonlinear_scalar = nonlinear_scalar_oracle
if 'scalar_terminal_hook' in mod.__dict__:
    _sth0 = mod.scalar_terminal_hook
    def scalar_terminal_hook_oracle(*a, **kw):
        out = _sth0(*a, **kw)
        if MODE.get('head', 1.0) != 1.0: MODE['log'].append('head x %.2f' % MODE['head']); return out * MODE['head']
        return out
    mod.scalar_terminal_hook = scalar_terminal_hook_oracle
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:      # the card pins these globals by identity; re-pin to the wrapped functions
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)

base = None
for s in settings:
    MODE.update(layers=set(int(x) for x in s.get('layers', [])), what=s.get('what', ''), strength=float(s.get('strength', 1.0)), log=[], head=float(s.get('head', 1.0)))
    t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(MLP(W, rank), 2**41), dtype=np.float64)
    y15 = y[15] if y.ndim == 2 else y; mse = float(np.mean((y15 - tp[15]) ** 2)); mall = float(np.mean((y - tp) ** 2)) if y.ndim == 2 else float('nan')
    if base is None: base = (mse, mall)
    if s.get('save'): np.savez_compressed('oracle_pred_rank%d_%s.npz' % (rank, s['save']), y15=y15, truth15=tp[15], y=y)
    print('rank %d %-52s FINAL %.4f e-9 vs base %+.2f%% | all-layer %+.2f%% | wall %.0fs | %s' % (rank, json.dumps(s), mse * 1e9, 100 * (mse / base[0] - 1), 100 * (mall / base[1] - 1), time.time() - t0, '; '.join(MODE['log'][:1]) + (' ... %d layers' % (len(MODE['log']) - 1) if len(MODE['log']) > 1 else '')), flush=True)
print('ORACLE_DONE', flush=True)
