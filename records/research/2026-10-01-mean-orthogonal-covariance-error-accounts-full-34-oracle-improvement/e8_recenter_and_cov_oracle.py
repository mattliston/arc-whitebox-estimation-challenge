#!/usr/bin/env python3
"""TAG opus-reexam E8: closed-loop tests of the carried covariance on the CURRENT torch card (construction copied from analyst A6).
(a) RECENTRING SCAN. After the learned mean correction delta the card shifts its covariance: C -= kappa (host d^T + d host^T + d d^T),
    kappa = 0.92 as deployed. E-anatomy on network 35 says the true covariance wants kappa of about 0.1. Arms rc<kappa>: same card,
    learned parts ON, no refit, only kappa changed.
(b) COVARIANCE ORACLES. With E = C_card - C_truth (32M-sample truth, fable_covhunter_v1/mcall) and mh = m/|m| at the same layer:
      alpha = mh'E mh, dperp = E mh - alpha mh,  E = alpha mh mh' + (mh dperp' + dperp mh') + Eperp.
    Arms subtract strength s x one part from the carried covariance at layers LO..14 (after recentring, hook 'cov_out'):
      mm (alpha mh mh'), cross (mh dperp' + dperp mh'), mal (both), perp (Eperp), full (E).
Target: organizers' 1e9-sample truth of the post-activation means (independent of the covariance truth).
usage: e8_recenter_and_cov_oracle.py <rank> <threads> [LO=2]  -> e8_rank{rank}.json"""
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
rank = int(sys.argv[1]); threads = int(sys.argv[2]); LO = int(sys.argv[3]) if len(sys.argv) > 3 else 2
arrays, dials = load_s33_arrays(np)
T = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in arrays.items()}
p = {key[2:]: value for key, value in T.items() if key.startswith('p.')}
e = {key[2:]: value for key, value in T.items() if key.startswith('e.')}
DL = {key: torch.tensor(value, dtype=torch.float64) for key, value in dials.items()}
spec = importlib.util.spec_from_file_location('e8_current_card', P2 / 'fable_physicist_v1/refit6b/s6bcard_loop5js.py')
LC = importlib.util.module_from_spec(spec); spec.loader.exec_module(LC)
assert LC.SCHED == '384-320-384'
bd_source = inspect.getsource(LC.JT.make_bd); assert bd_source.count('for j in range(4):') == 1
exec(compile(ast.parse(bd_source.replace('for j in range(4):', 'for j in range(3):')), str(__file__) + ':j3cut', 'exec'), LC.JT.__dict__)
nodes = [n for n in ast.parse((TR / 'train_segment.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'install_star']
assert len(nodes) == 1
ns = dict(torch=torch); exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TR / 'train_segment.py') + ':install_star', 'exec'), ns)
torch.set_num_threads(threads); torch.set_grad_enabled(False)
w = LC.JT.weights(rank); tp = LC.JT.truthpost(rank); LC.JT.DIALS_T = DL
mcp = P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'
mc = np.load(mcp) if mcp.exists() else None


def run(kappa=None, part=None, s=1.0, learned=True):
    cd = LC.card(torch, w, p, T['a'], T['c'], e, fine_ckpt=False)
    ns['install_star'](cd)
    if kappa is not None: cd.knobs['recenter'] = float(kappa)
    stats = {}
    if not learned:
        cd.hooks['delta'] = lambda layer, d, ctx: d if layer == 0 else torch.zeros_like(d)
    if part:
        def cov_out(layer, C, ctx):
            if layer < LO or layer > 14: return C
            Tc = torch.from_numpy(mc[f'cov_post_{layer}'].astype(np.float64))
            E = C - Tc; m = ctx['m_out']; mh = m / m.norm()
            al = mh @ E @ mh; dp = E @ mh - al * mh
            mm = al * torch.outer(mh, mh); cross = torch.outer(mh, dp) + torch.outer(dp, mh)
            piece = dict(mm=mm, cross=cross, mal=mm + cross, perp=E - mm - cross, full=E)[part]
            stats[layer] = float((piece ** 2).sum() / (E ** 2).sum())
            return C - s * piece
        cd.hooks['cov_out'] = cov_out
    pred = cd.run(False).numpy().astype(np.float64)
    return ((pred - tp) ** 2).mean(1), stats


plan = [('base', dict())]
plan += [(f'rc{k:g}', dict(kappa=k)) for k in (1.0, 0.6, 0.3, 0.1, 0.0)]
plan += [('learned_off', dict(learned=False))]
if mc is not None:
    plan += [(f'{part}_s{s:g}', dict(part=part, s=s)) for part in ('mm', 'cross', 'mal', 'perp', 'full') for s in (0.5, 1.0)]
    plan += [(f'rc0.1+{part}_s{s:g}', dict(kappa=0.1, part=part, s=s)) for part in ('cross',) for s in (0.5, 1.0)]
out = dict(TAG='opus-reexam E8 recentring scan + covariance-part oracles, current torch card', rank=rank, LO=LO, arms={})
t0 = time.time()
for name, kw in plan:
    mse, stats = run(**kw)
    out['arms'][name] = dict(final=float(mse[15]), per_layer=[float(x) for x in mse], part_share=stats, **{k: v for k, v in kw.items()})
    print(json.dumps(dict(rank=rank, arm=name, final_x1e9=round(float(mse[15]) * 1e9, 3),
                          vs_base_pct=round((float(mse[15]) / out['arms']['base']['final'] - 1) * 100, 2),
                          all_layer_x1e9=round(float(mse.mean()) * 1e9, 3), wall=round(time.time() - t0, 1))), flush=True)
json.dump(out, open(HERE / f'e8_rank{rank:04d}.json', 'w'), indent=1)
print(json.dumps(dict(done=True, rank=rank)), flush=True)
