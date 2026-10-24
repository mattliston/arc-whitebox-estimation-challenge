#!/usr/bin/env python3
"""TAG analyst A2: dump the CURRENT torch card's per-unit pre-activation cumulants at every layer (learned ON, closed loop,
forward only, torch.no_grad). The card is fable_physicist_v1/refit6b/s6bcard_loop5js.py built exactly as
H/hill_climb_rrpi_detached_star_v1/train_segment.py builds it: env JOIN_SCHED=384-320-384, the j3cut exec patch of JT.make_bd,
the trainer's install_star (compiled from the trainer's own source text, not retyped: the trainer module runs on import),
the k4_15 hook set to identity, parameters = native_parameters.load_s33_arrays (the 28 literal arrays and 12 dials of the research
parent H/hill_climb_pair_rr_inverse_v1/candidate.py, sha a3efb3a1...), dials through LC.JT.DIALS_T.
Captured per layer and per unit (hooks 'pair_in', 'k3_15', 'k4_15', 'delta', 'cov_out'; each wraps the hook already installed):
  mu, v, k3 (and its parts: bank diagonal, star diagonal, gain term 3 v b), k4 (after the k4 band dial), radial, host mean,
  learned correction delta, diagonal of the carried post covariance, alive / response masks; pred (16,1024) and the organizer truth.
Optional third argument: comma list of layers whose full carried post covariance (hook 'cov_out') is saved to a2_cov_rank{rank}.npz.
usage: a2_dump_current_card.py <rank> <threads> [cov layers, e.g. 6,10,13]  -> a2_card_rank{rank:04d}.npz"""
import os, sys, ast, json, time, inspect, resource, importlib.util
from pathlib import Path
os.environ['JOIN_SCHED'] = '384-320-384'
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; H = P2 / 'codex_resume_20260926_v2'
TR = H / 'hill_climb_rrpi_detached_star_v1'
for rel in ('fable_analyst_v1', 'fable_covhunter_v1', 'fable_engineer_v1', 'fable_physicist_v1/refit6b', 'fable_porter_v1'):
    sys.path.insert(0, str(P2 / rel))
sys.path.insert(0, str(TR))
import numpy as np, torch
from native_parameters import load_s33_arrays, SOURCE_SHA
rank = int(sys.argv[1]); threads = int(sys.argv[2])
cov_layers = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else []
arrays, dials = load_s33_arrays(np)
T = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in arrays.items()}
p = {key[2:]: value for key, value in T.items() if key.startswith('p.')}
e = {key[2:]: value for key, value in T.items() if key.startswith('e.')}
assert len(e) == 11 and len(T) == 28
DL = {key: torch.tensor(value, dtype=torch.float64) for key, value in dials.items()}
spec = importlib.util.spec_from_file_location('a2_current_card', P2 / 'fable_physicist_v1/refit6b/s6bcard_loop5js.py')
LC = importlib.util.module_from_spec(spec); spec.loader.exec_module(LC)
assert LC.SCHED == '384-320-384'
bd_source = inspect.getsource(LC.JT.make_bd)
assert bd_source.count('for j in range(4):') == 1
exec(compile(ast.parse(bd_source.replace('for j in range(4):', 'for j in range(3):')), str(__file__) + ':j3cut', 'exec'), LC.JT.__dict__)
trainer_text = (TR / 'train_segment.py').read_text()
nodes = [n for n in ast.parse(trainer_text).body if isinstance(n, ast.FunctionDef) and n.name == 'install_star']
assert len(nodes) == 1
ns = dict(torch=torch); exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TR / 'train_segment.py') + ':install_star', 'exec'), ns)
torch.set_num_threads(threads); torch.set_grad_enabled(False)

w = LC.JT.weights(rank); tp = LC.JT.truthpost(rank)
assert w.shape == (16, 1024, 1024) and tp.shape == (16, 1024)
LC.JT.DIALS_T = DL
cd = LC.card(torch, w, p, T['a'], T['c'], e, fine_ckpt=False)
cd.hooks['k4_15'] = lambda layer, value, ctx: value
ns['install_star'](cd)
old = dict(cd.hooks); cap = {}; cov = {}
def npy(x): return x.detach().double().numpy().copy()
def pair_in(layer, value, ctx):
    if old.get('pair_in') is not None: value = old['pair_in'](layer, value, ctx)
    mu, Cpre, k21, k4, k22, rem = value
    vv = torch.diagonal(Cpre)
    cap[f'mu_{layer}'] = npy(mu); cap[f'v_{layer}'] = npy(vv); cap[f'k3_{layer}'] = npy(torch.diagonal(k21))
    cap[f'k3bank_{layer}'] = npy(torch.diagonal(ctx['cp'])); cap[f'k3star_{layer}'] = npy(torch.diagonal(ctx['sp']))
    cap[f'k3b_{layer}'] = npy(3 * vv * ctx['bpre']); cap[f'bpre_{layer}'] = npy(ctx['bpre'])
    cap[f'k4_{layer}'] = npy(k4 if k4.ndim else k4.expand(1024)); cap[f'radial_{layer}'] = npy(ctx['radial'])
    cap[f'alive_{layer}'] = npy(ctx['alive']); cap[f'resp_{layer}'] = npy(ctx['response_mask'])
    return value
def k3_15(layer, k3, ctx):
    k3n = old['k3_15'](layer, k3, ctx) if old.get('k3_15') is not None else k3
    cap['mu_15'] = npy(ctx['mu']); cap['v_15'] = npy(ctx['vv']); cap['k3_15'] = npy(k3n); cap['k3raw_15'] = npy(k3)
    cap['k3bank_15'] = npy(ctx['bank']); cap['k3star_15'] = npy(ctx['star']); cap['k3b_15'] = npy(ctx['bterm']); cap['radial_15'] = npy(ctx['radial'])
    return k3n
def k4_15(layer, k4, ctx):
    k4n = old['k4_15'](layer, k4, ctx)
    cap['k4_15'] = npy(k4n)
    return k4n
def delta(layer, d, ctx):
    if old.get('delta') is not None: d = old['delta'](layer, d, ctx)
    cap[f'host_{layer}'] = npy(ctx['host']); cap[f'delta_{layer}'] = npy(d)
    return d
def cov_out(layer, C, ctx):
    if old.get('cov_out') is not None: C = old['cov_out'](layer, C, ctx)
    cap[f'cpost_diag_{layer}'] = npy(torch.diagonal(C))
    if layer in cov_layers: cov[f'cov_out_{layer}'] = npy(C); cov[f'm_out_{layer}'] = npy(ctx['m_out'])
    return C
cd.hooks.update(pair_in=pair_in, k3_15=k3_15, k4_15=k4_15, delta=delta, cov_out=cov_out)
t0 = time.time()
with torch.no_grad(): pred = cd.run(False).numpy().astype(np.float64)
wall = time.time() - t0
cap['pred'] = pred; cap['truth'] = tp
mse = ((pred - tp) ** 2).mean(1)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
print(json.dumps(dict(TAG='analyst A2 current torch card (s6bcard_loop5js as the trainer builds it), learned ON, forward only',
                      rank=rank, threads=threads, source_sha=SOURCE_SHA, dials=dials, hooks_wrapped=sorted(k for k in old),
                      per_layer_mse_x1e9=[round(float(x) * 1e9, 3) for x in mse], final_mse=float(mse[15]), all_layer_mse=float(mse.mean()),
                      wall_s=round(wall, 1), peak_rss_gb=round(rss, 2))), flush=True)
np.savez(HERE / f'a2_card_rank{rank:04d}.npz', **cap)
if cov: np.savez(HERE / f'a2_cov_rank{rank:04d}.npz', **cov)
print(json.dumps(dict(done=True, rank=rank, out=str(HERE / f'a2_card_rank{rank:04d}.npz'))), flush=True)
