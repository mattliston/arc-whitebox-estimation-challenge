#!/usr/bin/env python3
"""TAG opus-reexam E8b: recentring-strength scan on many networks (current torch card, learned parts ON, no refit).
After the learned mean correction d the card shifts its covariance by -kappa (host d' + d host' + d d'); deployed kappa = 0.92.
usage: e8b_recenter_scan.py <threads> <kappas comma list> <ranks comma list | valid40> <out tag>  -> e8b_<tag>.jsonl (one row per net)"""
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
threads = int(sys.argv[1]); kappas = [float(x) for x in sys.argv[2].split(',')]
ranks = json.loads((P2 / 'fable_analyst_v1/split_train640.json').read_text())['valid40'] if sys.argv[3] == 'valid40' else [int(x) for x in sys.argv[3].split(',')]
tag = sys.argv[4]
arrays, dials = load_s33_arrays(np)
T = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in arrays.items()}
p = {key[2:]: value for key, value in T.items() if key.startswith('p.')}
e = {key[2:]: value for key, value in T.items() if key.startswith('e.')}
DL = {key: torch.tensor(value, dtype=torch.float64) for key, value in dials.items()}
spec = importlib.util.spec_from_file_location('e8b_current_card', P2 / 'fable_physicist_v1/refit6b/s6bcard_loop5js.py')
LC = importlib.util.module_from_spec(spec); spec.loader.exec_module(LC)
assert LC.SCHED == '384-320-384'
bd_source = inspect.getsource(LC.JT.make_bd); assert bd_source.count('for j in range(4):') == 1
exec(compile(ast.parse(bd_source.replace('for j in range(4):', 'for j in range(3):')), str(__file__) + ':j3cut', 'exec'), LC.JT.__dict__)
nodes = [n for n in ast.parse((TR / 'train_segment.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'install_star']
ns = dict(torch=torch); exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TR / 'train_segment.py') + ':install_star', 'exec'), ns)
torch.set_num_threads(threads); torch.set_grad_enabled(False); LC.JT.DIALS_T = DL
out = open(HERE / f'e8b_{tag}.jsonl', 'a'); t0 = time.time()
for rank in ranks:
    w = LC.JT.weights(rank)
    tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    row = dict(rank=rank, final={}, all_layer={})
    for k in kappas:
        cd = LC.card(torch, w, p, T['a'], T['c'], e, fine_ckpt=False); ns['install_star'](cd)
        cd.knobs['recenter'] = k
        mse = ((cd.run(False).numpy().astype(np.float64) - tp) ** 2).mean(1)
        row['final'][f'{k:g}'] = float(mse[15]); row['all_layer'][f'{k:g}'] = float(mse.mean())
    row['wall'] = round(time.time() - t0, 1)
    out.write(json.dumps(row) + '\n'); out.flush(); print(json.dumps(row), flush=True)
