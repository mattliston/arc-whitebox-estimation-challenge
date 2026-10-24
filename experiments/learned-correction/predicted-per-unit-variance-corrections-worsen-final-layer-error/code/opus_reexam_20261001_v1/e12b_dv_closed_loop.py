#!/usr/bin/env python3
"""TAG opus-reexam E12b: does a PREDICTED per-unit variance correction help in closed loop?
E12 found the card's relative per-unit variance error 10-20% predictable (held-out R2) from the learned net's own inputs.
Here a predictor (ridge and gradient boosting per layer, trained on the two OTHER networks with Monte Carlo truth) supplies
v' = v_card / (1 + yhat) to the mean formula of every layer 4..15 of the test network (the shift goes through the 'delta' hook as in
analyst A6: the corrected mean is carried forward; covariance, bank and learned parts keep their own inputs).
Features are taken from the e7 dump of the UNMODIFIED run of the test network (open-loop features).
Arms: base, predicted (gbm), predicted (ridge), predicted x0.5, exact truth (the A6 'v' arm, for scale); each with the learned mean
correction on and off.
usage: e12b_dv_closed_loop.py <test rank> <threads>"""
import os, sys, ast, json, time, inspect, importlib.util
from pathlib import Path
os.environ['JOIN_SCHED'] = '384-320-384'
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; H = P2 / 'codex_resume_20260926_v2'
TR = H / 'hill_climb_rrpi_detached_star_v1'
for rel in ('fable_analyst_v1', 'fable_covhunter_v1', 'fable_engineer_v1', 'fable_physicist_v1/refit6b', 'fable_porter_v1'):
    sys.path.insert(0, str(P2 / rel))
sys.path.insert(0, str(TR))
import numpy as np, torch, lightgbm as lgb
from native_parameters import load_s33_arrays
test = int(sys.argv[1]); threads = int(sys.argv[2])
G = P2 / 'public_g19_attribution_v1'
def truth_v(rank):
    mc = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); L = 16
    vB = np.array([np.diag(mc[f'cov_pre_{l}']) for l in range(L)]).astype(np.float64); NB = float(mc['P'])
    if (G / f'mps_truth_rank{rank:04d}_P32M.npz').exists():
        a = np.load(G / f'mps_truth_rank{rank:04d}_P32M.npz'); b = np.load(G / f'mps_truth_rank{rank:04d}_indep32M.npz')
        return (a['mc_pre_v'].astype(np.float64) * float(a['P']) + b['mc_pre_v'].astype(np.float64) * float(b['P']) + vB * NB) / (float(a['P']) + float(b['P']) + NB)
    d = np.load(HERE / f'analyst/e1_rank{rank:04d}_N16M_B16.npz'); N = float(d['N']); vA = []
    for l in range(L):
        tot = d[f'norm_{l}_cnt'].sum(); m1 = d[f'norm_{l}_0'].sum(0) / tot; m2 = d[f'norm_{l}_1'].sum(0) / tot; vA.append(m2 - m1 ** 2)
    return (np.array(vA) * N + vB * NB) / (N + NB)
phi = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
def feats(d, l):
    v = d[f'v_{l}'].astype(np.float64); mu = d[f'mu_{l}'].astype(np.float64); s = np.sqrt(v); a = mu / s
    extra = np.stack([a, np.abs(a), phi(a), a * phi(a), d[f'k3bank_{l}'] / s ** 3, d[f'k3star_{l}'] / s ** 3, d[f'k3b_{l}'] / s ** 3, np.log(np.maximum(v, 1e-12))], 1)
    return np.concatenate([d[f'x_{l}'], d[f'hs_{l}'], extra], 1).astype(np.float32), v
ranks = [35, 37, 39]; train = [r for r in ranks if r != test]
dumps = {r: np.load(HERE / f'e7/e7_rank{r:04d}.npz') for r in ranks}; vt = {r: truth_v(r) for r in ranks}
pred = {'gbm': {}, 'ridge': {}}
for l in range(4, 16):
    Xa, ya = [], []
    for r in train:
        X, v = feats(dumps[r], l); ok = vt[r][l] > 1e-8; Xa.append(X[ok]); ya.append(((v - vt[r][l]) / np.where(ok, vt[r][l], 1.0))[ok])
    Xa = np.concatenate(Xa); ya = np.concatenate(ya); Xb, vb = feats(dumps[test], l)
    m, sd = Xa.mean(0), Xa.std(0) + 1e-9; A = (Xa - m) / sd; B = (Xb - m) / sd
    coef = np.linalg.solve(A.T @ A + 1e-1 * len(A) * np.eye(A.shape[1]), A.T @ (ya - ya.mean())); lo, hi = np.percentile(ya, [0.5, 99.5])
    pred['ridge'][l] = np.clip(B @ coef + ya.mean(), lo, hi)
    idx = np.random.default_rng(0).permutation(len(Xa)); cut = int(len(Xa) * 0.8)
    mdl = lgb.train(dict(objective='regression', learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.7, lambda_l2=1.0, verbose=-1, num_threads=threads),
                    lgb.Dataset(Xa[idx[:cut]], ya[idx[:cut]]), num_boost_round=400, valid_sets=[lgb.Dataset(Xa[idx[cut:]], ya[idx[cut:]])],
                    callbacks=[lgb.early_stopping(30, verbose=False)])
    pred['gbm'][l] = mdl.predict(Xb, num_iteration=mdl.best_iteration)
arrays, dials = load_s33_arrays(np)
T = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in arrays.items()}
p = {key[2:]: value for key, value in T.items() if key.startswith('p.')}
e = {key[2:]: value for key, value in T.items() if key.startswith('e.')}
spec = importlib.util.spec_from_file_location('e12b_current_card', P2 / 'fable_physicist_v1/refit6b/s6bcard_loop5js.py')
LC = importlib.util.module_from_spec(spec); spec.loader.exec_module(LC)
bd_source = inspect.getsource(LC.JT.make_bd); assert bd_source.count('for j in range(4):') == 1
exec(compile(ast.parse(bd_source.replace('for j in range(4):', 'for j in range(3):')), str(__file__) + ':j3cut', 'exec'), LC.JT.__dict__)
nodes = [n for n in ast.parse((TR / 'train_segment.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'install_star']
ns = dict(torch=torch); exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TR / 'train_segment.py') + ':install_star', 'exec'), ns)
torch.set_num_threads(threads); torch.set_grad_enabled(False)
LC.JT.DIALS_T = {k: torch.tensor(v, dtype=torch.float64) for k, v in dials.items()}
w = LC.JT.weights(test); tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{test:04d}.npz')['mc_post'].astype(np.float64)
def run(mode, scale, learned):
    cd = LC.card(torch, w, p, T['a'], T['c'], e, fine_ckpt=False)
    stash = {}
    def k4_15(layer, value, ctx): stash['k4'] = value; return value
    cd.hooks['k4_15'] = k4_15
    ns['install_star'](cd)
    old_k3 = cd.hooks.get('k3_15')
    def k3_15(layer, k3, ctx):
        k3n = old_k3(layer, k3, ctx) if old_k3 is not None else k3
        stash['k3'] = k3n; return k3n
    cd.hooks['k3_15'] = k3_15
    NS = cd.env['nonlinear_scalar']
    def delta(layer, d, ctx):
        keep = d if (learned or layer == 0) else torch.zeros_like(d)
        if mode == 'base' or layer < 4: return keep
        host, mu = ctx['host'], ctx['mu']
        if layer == 15: v, k3, k4 = ctx['vv'], stash['k3'], stash['k4']
        else: v = torch.diagonal(ctx['Cpre']); k3 = torch.diagonal(ctx['k21']); k4 = ctx['k4']
        k4 = k4 if k4.ndim else k4.expand(1024)
        if mode == 'truth': v2 = torch.from_numpy(vt[test][layer])
        else: v2 = v / (1 + scale * torch.from_numpy(pred[mode][layer].astype(np.float64)))
        g1 = k3 / v ** 1.5; g2 = k4 / v ** 2
        return keep + (NS(mu, v2, g1 * v2 ** 1.5, g2 * v2 ** 2)[0] - host)
    cd.hooks['delta'] = delta
    out = cd.run(False).numpy().astype(np.float64)
    return ((out - tp) ** 2).mean(1)
t0 = time.time(); base = {}
for learned in (True, False):
    for name, mode, scale in (('base', 'base', 0), ('predicted gbm', 'gbm', 1.0), ('predicted gbm x0.5', 'gbm', 0.5), ('predicted ridge', 'ridge', 1.0), ('exact truth (layers 4-15)', 'truth', 1.0)):
        mse = run(mode, scale, learned)
        if name == 'base': base[learned] = mse[15]
        print(json.dumps(dict(test_rank=test, learned=learned, arm=name, final_x1e9=round(float(mse[15]) * 1e9, 3), vs_base_pct=round((float(mse[15]) / base[learned] - 1) * 100, 2), wall=round(time.time() - t0, 1))), flush=True)
