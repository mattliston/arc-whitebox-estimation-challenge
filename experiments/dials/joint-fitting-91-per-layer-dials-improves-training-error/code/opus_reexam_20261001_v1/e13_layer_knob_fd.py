#!/usr/bin/env python3
"""TAG opus-reexam E13: finite-difference Jacobian of the final-layer error with respect to PER-LAYER physics scalars
(current torch card, learned parts ON, no refit), for a Gauss-Newton fit of per-layer dials (e13_fit.py).
One run per (family, layer): the scalar of that family is multiplied by (1 + step) at that layer only; the change of the final-layer
error vector divided by the step is one Jacobian column. Families:
  cov    strength of the covariance-output knob delta (iso / k21n / k31 of gn3 V3 together): C_plain + s (C_knob - C_plain), layers 2-14
  edge   learned covariance edge increment, layers 8-14
  k4     per-unit fourth cumulant fed to the pair map and the mean formula, layers 1-15
  b      the Sym(C x b) carrier after its recursion, layers 1-14
  carry  weights of the carried bank columns, layers 2-14;    new   weights of the newborn columns, layers 0-14
  rc     recentring strength after the learned mean correction, layers 1-14
usage: e13_layer_knob_fd.py <ranks comma list> <threads> <tag>  -> e13_<tag>.npz (per rank: e0, U [1024 x n_knobs]; knob names)"""
import os, sys, ast, json, time, inspect, importlib.util
from pathlib import Path
os.environ['JOIN_SCHED'] = '384-320-384'
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; H = P2 / 'codex_resume_20260926_v2'
TR = H / 'hill_climb_rrpi_detached_star_v1'
for rel in ('fable_analyst_v1', 'fable_covhunter_v1', 'fable_engineer_v1', 'fable_physicist_v1/refit6b', 'fable_porter_v1'):
    sys.path.insert(0, str(P2 / rel))
sys.path.insert(0, str(TR))
import numpy as np, torch
from native_parameters import load_s33_arrays
ranks = [int(x) for x in sys.argv[1].split(',')]; threads = int(sys.argv[2]); tag = sys.argv[3]
arrays, dials = load_s33_arrays(np)
T = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in arrays.items()}
p = {key[2:]: value for key, value in T.items() if key.startswith('p.')}
e = {key[2:]: value for key, value in T.items() if key.startswith('e.')}
spec = importlib.util.spec_from_file_location('e13_current_card', P2 / 'fable_physicist_v1/refit6b/s6bcard_loop5js.py')
LC = importlib.util.module_from_spec(spec); spec.loader.exec_module(LC)
bd_source = inspect.getsource(LC.JT.make_bd); assert bd_source.count('for j in range(4):') == 1
exec(compile(ast.parse(bd_source.replace('for j in range(4):', 'for j in range(3):')), str(__file__) + ':j3cut', 'exec'), LC.JT.__dict__)
nodes = [n for n in ast.parse((TR / 'train_segment.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'install_star']
ns = dict(torch=torch); exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TR / 'train_segment.py') + ':install_star', 'exec'), ns)
torch.set_num_threads(threads); torch.set_grad_enabled(False)
LC.JT.DIALS_T = {k: torch.tensor(v, dtype=torch.float64) for k, v in dials.items()}
STEP = dict(cov=0.5, edge=0.3, k4=0.05, b=0.05, carry=0.01, new=0.02, rc=-0.3)
LAYERS = dict(cov=range(2, 15), edge=range(8, 15), k4=range(1, 16), b=range(1, 15), carry=range(2, 15), new=range(0, 15), rc=range(1, 15))
KNOBS = [(f, l) for f in STEP for l in LAYERS[f]]


def run(w, tp, scale):
    """scale: dict (family, layer) -> multiplicative factor (default 1)."""
    g = lambda f, l: scale.get((f, l), 1.0)
    cd = LC.card(torch, w, p, T['a'], T['c'], e, fine_ckpt=False); ns['install_star'](cd)
    h = dict(cd.hooks)
    if 'cov_pm' in h:
        cd.hooks['cov_pm'] = lambda l, C, ctx: C + g('cov', l) * (h['cov_pm'](l, C, ctx) - C)
    old_edge = h.get('edge')
    cd.hooks['edge'] = lambda l, inc, ctx: (old_edge(l, inc, ctx) if old_edge else inc) * g('edge', l)
    old_pin = h.get('pair_in')
    def pair_in(l, pin, ctx):
        pin = old_pin(l, pin, ctx) if old_pin else pin
        s = g('k4', l)
        if s != 1.0:
            ctx['k4'] = ctx['k4'] * s
            pin = (pin[0], pin[1], pin[2], pin[3] * s, pin[4], pin[5])
        return pin
    cd.hooks['pair_in'] = pair_in
    old_k415 = h.get('k4_15')
    cd.hooks['k4_15'] = lambda l, k4, ctx: (old_k415(l, k4, ctx) if old_k415 else k4) * g('k4', 15)
    old_b = h.get('b_post')
    cd.hooks['b_post'] = lambda l, b, ctx: (old_b(l, b, ctx) if old_b else b) * g('b', l)
    old_so = h.get('state_out')
    def state_out(l, st, ctx):
        st = old_so(l, st, ctx) if old_so else st
        if l == 15: return st
        sc, sn = g('carry', l), g('new', l)
        if sc == 1.0 and sn == 1.0: return st
        lam, stops = st[3], st[12]; n_old = stops[-2] if len(stops) >= 2 else 0
        return st[:3] + (torch.cat((lam[:n_old] * sc, lam[n_old:] * sn)),) + st[4:]
    cd.hooks['state_out'] = state_out
    old_delta = h.get('delta'); base_rc = float(cd.knobs['recenter'])
    def delta(l, d, ctx):
        d = old_delta(l, d, ctx) if old_delta else d
        cd.knobs['recenter'] = base_rc * g('rc', l)
        return d
    cd.hooks['delta'] = delta
    pred = cd.run(False).numpy().astype(np.float64)
    return pred[15] - tp[15], float(((pred - tp) ** 2).mean(1)[15])


out = dict(knobs=np.array([f'{f}:{l}' for f, l in KNOBS]), steps=np.array([STEP[f] for f, l in KNOBS]), ranks=np.array(ranks)); t0 = time.time()
for rank in ranks:
    w = LC.JT.weights(rank); tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    e0, m0 = run(w, tp, {})
    cols = []
    for (f, l) in KNOBS:
        e1, m1 = run(w, tp, {(f, l): 1.0 + STEP[f]})
        cols.append((e1 - e0) / STEP[f])
        print(json.dumps(dict(rank=rank, knob=f'{f}:{l}', mse_ratio=round(m1 / m0, 5), wall=round(time.time() - t0, 1))), flush=True)
    out[f'e0_{rank}'] = e0; out[f'U_{rank}'] = np.stack(cols, 1)
    np.savez(HERE / f'e13_{tag}.npz', **out)
    print(json.dumps(dict(rank=rank, base_x1e9=round(m0 * 1e9, 4), done=True, wall=round(time.time() - t0, 1))), flush=True)
