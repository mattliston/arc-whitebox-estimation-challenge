"""TAG fable-lead 2026-10-20: NATIVE truth-plane oracle. Runs the shipped card in-process on a store network (35 / 100, Monte Carlo planes of
public_g19_attribution_v1) and adds, at chosen layers, the truth's K31 REMAINDER (the part of T31_true = K31/(sigma_i^3 sigma_j) not proportional,
row by row, to rho) to the covariance through the card's own GN kernel (_gn_dcov: X = D * t31a / 6, dcov = (X + X') * ss). Base first.
usage: native_k31_oracle_v1.py <candidate> <rank> <settings json: [{"layers": [..], "strength": x, "mode": "rest"|"full"}, ...]>"""
import json, sys, time, types
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
class pc:      # the torch card's loaders (fable_covhunter_v1/covstep.py, pcard.py) without torch
    weights = staticmethod(lambda rank: np.load(STORE / Path(MAN[rank]['path']).parent / 'weights.npy'))
    truth = staticmethod(lambda rank: np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64))
    planes = staticmethod(lambda rank: np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}.npz'))
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
W = pc.weights(rank); tp = pc.truth(rank); PLN = pc.planes(rank)
W = [np.asarray(w.numpy() if hasattr(w, 'numpy') else w) for w in (W if isinstance(W, (list, tuple)) else [W[l] for l in range(16)])]
print('rank', rank, 'weights', len(W), W[0].shape, W[0].dtype, 'truth', tp.shape, 'planes', [k for k in PLN.files if k.startswith('K31')][:3], flush=True)


class MLP:
    def __init__(self, weights, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'store-%d' % seed


MODE = dict(layers=set(), strength=0.0, mode='rest', diag=[])
orig = mod._gn_dcov
def oracle_dcov(l, mu, Cpre, K21, k4, aux):
    out = orig(l, mu, Cpre, K21, k4, aux)
    if int(l) not in MODE['layers'] or not MODE['strength'] or ('K31_%d' % int(l)) not in PLN.files:
        return out
    n = 1024; eye = np.eye(n, dtype=bool)
    C = np.asarray(Cpre, dtype=np.float64); v = np.diag(C).copy(); s = np.sqrt(v); ss = np.outer(s, s); rho = C / ss; rho[eye] = 0.0
    T = np.asarray(PLN['K31_%d' % int(l)], dtype=np.float64) / (v[:, None] * s[:, None] * s[None, :]); T[eye] = 0.0
    a_true = (T * rho).sum(1) / np.maximum((rho * rho).sum(1), 1e-300)
    D = T - a_true[:, None] * rho if MODE['mode'] == 'rest' else T
    D[eye] = 0.0
    if MODE['mode'] in ('k21', 'both', 'k21col', 'k21rest', 'k21scale', 'k21rowscale', 'k21dump', 'k21rho3'):      # the third-order plane: truth K21 minus the card's own plane (the function's K21 argument), off the diagonal, through dT21 * t21a / 2
        K21c = np.asarray(K21, dtype=np.float64); D21 = np.asarray(PLN['K21_%d' % int(l)], dtype=np.float64) - K21c; D21[eye] = 0.0
        MODE['diag'].append((int(l), 'K21 card/truth norm %.3f cos %.3f' % (np.linalg.norm(K21c - np.diag(np.diag(K21c))) / np.linalg.norm(np.asarray(PLN['K21_%d' % int(l)]) - np.diag(np.diag(np.asarray(PLN['K21_%d' % int(l)])))), float(((K21c - np.diag(np.diag(K21c))) * (np.asarray(PLN['K21_%d' % int(l)]) - np.diag(np.diag(np.asarray(PLN['K21_%d' % int(l)]))))).sum() / (np.linalg.norm(K21c - np.diag(np.diag(K21c))) * np.linalg.norm(np.asarray(PLN['K21_%d' % int(l)]) - np.diag(np.diag(np.asarray(PLN['K21_%d' % int(l)])))) + 1e-300)))))
        if MODE['mode'] == 'k21rho3':      # NO truth: the card's OWN K21 plane through the NEXT term of the GN kernel series (rho^3/6 outer(A5, A4)): is the kernel's rho-order a lever?
            D21 = np.where(~eye, K21c, 0.0); MODE['rho3'] = True
        if MODE['mode'] != 'both': D = D * 0.0
        mu_ = np.asarray(mu, dtype=np.float64); offd = ~eye
        if MODE['mode'] in ('k21col', 'k21rest'):      # the scale-mode column: row by row the projection of D on the column vector mu (q_i mu_j), off the diagonal
            M = np.where(offd, mu_[None, :], 0.0); q = (D21 * M).sum(1) / np.maximum((M * M).sum(1), 1e-300); P = q[:, None] * M
            MODE['diag'].append((int(l), 'col share %.3f' % (np.linalg.norm(P) ** 2 / np.linalg.norm(D21) ** 2))); D21 = P if MODE['mode'] == 'k21col' else D21 - P
        if MODE['mode'] == 'k21scale':      # one scale per layer along the card's own plane
            Kc = np.where(offd, K21c, 0.0); al = (D21 * Kc).sum() / max((Kc * Kc).sum(), 1e-300); MODE['diag'].append((int(l), 'scale alpha %.3f share %.3f' % (al, al * al * (Kc * Kc).sum() / (D21 * D21).sum()))); D21 = al * Kc
        if MODE['mode'] == 'k21rowscale':      # one scale per row along the card's own plane
            Kc = np.where(offd, K21c, 0.0); al = (D21 * Kc).sum(1) / np.maximum((Kc * Kc).sum(1), 1e-300); P = al[:, None] * Kc; MODE['diag'].append((int(l), 'rowscale mean %.3f share %.3f' % (al.mean(), np.linalg.norm(P) ** 2 / np.linalg.norm(D21) ** 2))); D21 = P
        if MODE['mode'] == 'k21dump':      # dump the planes and per-unit inputs of this layer for the offline fit, then no change
            np.savez_compressed('k21dump_rank%d_L%d.npz' % (rank, int(l)), K21_true=np.asarray(PLN['K21_%d' % int(l)], dtype=np.float32), K21_card=K21c.astype(np.float32), Cpre=C.astype(np.float32), mu=mu_, v=v, k3=np.diag(K21c).copy(), k4=np.asarray(k4, dtype=np.float64) * np.ones(n), alpha=np.asarray(aux['alpha'], dtype=np.float64), gate=np.asarray(aux['gate'], dtype=np.float64), pdf=np.asarray(aux['pdf'], dtype=np.float64), gmean=np.asarray(aux['gaussian_mean'], dtype=np.float64), K22_true=np.asarray(PLN['K22_%d' % int(l)], dtype=np.float32) if ('K22_%d' % int(l)) in PLN.files else np.zeros(1, np.float32), K31_true=np.asarray(PLN['K31_%d' % int(l)], dtype=np.float32), rest31=D.astype(np.float32))
            D21 = D21 * 0.0; D = D * 0.0
        dT21 = D21 / (v[:, None] * s[None, :])
    else: dT21 = None
    a = np.asarray(aux['alpha'], dtype=np.float64); s_a = np.asarray(aux['sigma'], dtype=np.float64)
    A, _u = mod.derivatives(aux['alpha'], aux['pdf'], aux['gate'], aux['gaussian_mean'] / aux['sigma'], 5)
    A1, A2, A3, A4, A5 = [np.asarray(x, dtype=np.float64) for x in A[1:6]]
    t31a = np.outer(A3, A1) + rho * np.outer(A4, A2) + (rho * rho / 2) * np.outer(A5, A3)
    if MODE['mode'] == 'k21dump':      # kernel rows for the weighted (effect-metric) analysis offline
        t21a_ = np.outer(A2, A1) + rho * np.outer(A3, A2) + (rho * rho / 2) * np.outer(A4, A3)
        np.savez_compressed('k21kern_rank%d_L%d.npz' % (rank, int(l)), A1=A1, A2=A2, A3=A3, A4=A4, A5=A5, s=s, t21a=t21a_.astype(np.float32), t31a=t31a.astype(np.float32))
    X = MODE['strength'] * D * t31a / 6.0
    if dT21 is not None:
        t21a = (rho ** 3 / 6) * np.outer(A5, A4) if MODE.get('rho3') else (np.outer(A2, A1) + rho * np.outer(A3, A2) + (rho * rho / 2) * np.outer(A4, A3)); X = X + MODE['strength'] * dT21 * t21a / 2.0
        if MODE.get('rho3'): MODE['diag'].append((int(l), 'rho3 term rms / 3-term rms %.4f' % (np.sqrt(((dT21 * t21a) ** 2).mean()) / np.sqrt(((dT21 * (np.outer(A2, A1) + rho * np.outer(A3, A2) + (rho * rho / 2) * np.outer(A4, A3))) ** 2).mean()))))
    dcov = (X + X.T) * ss; dcov[eye] = 0.0
    # diagnostic: the card's own row coefficient (k4 / v^2 x row scale x (1 + knob)) against the truth's
    try:
        th = mod._pld_knobs(l); k4v = np.asarray(k4, dtype=np.float64) * np.ones(n); t4 = k4v / v ** 2
        k3 = np.asarray(aux['pre_k3'], dtype=np.float64) if aux.get('pre_k3') is not None else np.diag(np.asarray(K21, dtype=np.float64))
        r31 = mod._k31_row_scale_gn(mu, k3, v) if getattr(mod, '_S19_GN_TORCH', False) else mod._k31_row_scale(mu, k3, v)
        a_card = t4 * (np.asarray(r31, dtype=np.float64) if r31 is not None else 1.0) * (1 + float(th['k31']))
        MODE['diag'].append((int(l), round(float(np.corrcoef(a_true, a_card)[0, 1]), 3), round(float(a_card.mean() / a_true.mean()), 3), round(float(np.linalg.norm(D) / np.linalg.norm(T)), 3), round(float(np.abs(dcov).max()), 6)))
    except Exception as e:
        MODE['diag'].append((int(l), 'diag failed: %r' % (e,)))
    return out + fnp.asarray(dcov) if hasattr(fnp, 'asarray') else out + fnp.array(dcov)
mod._gn_dcov = oracle_dcov

base = None
for st in settings:
    MODE.update(layers=set(int(x) for x in st.get('layers', [])), strength=float(st.get('strength', 1.0)), mode=st.get('mode', 'rest'), diag=[], rho3=False)
    t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(MLP(W, rank), 2**41), dtype=np.float64)
    y15 = y[15] if y.ndim == 2 else y; mse = float(np.mean((y15 - tp[15]) ** 2)); mall = float(np.mean((y - tp) ** 2)) if y.ndim == 2 else float('nan')
    if base is None: base = (mse, mall)
    print('rank %d %-40s FINAL %.4f e-9  vs base %+.2f%%  | all-layer %.4f e-9 %+.2f%%  wall %.0fs' % (rank, json.dumps({k: st[k] for k in st if k != 'diag'}), mse * 1e9, 100 * (mse / base[0] - 1), mall * 1e9, 100 * (mall / base[1] - 1), time.time() - t0), flush=True)
    for d in MODE['diag']: print(('    layer %s row_corr %s card/truth %s rest/whole %s max|dcov| %s' % d) if len(d) == 5 else ('    layer %s %s' % d), flush=True)
print('ORACLE_DONE', flush=True)
