"""TAG opus-physicist: the CURRENT torch research card, built exactly as H/hill_climb_rrpi_detached_star_v1/train_segment.py builds it
(construction copied from O/analyst/a2_dump_current_card.py and O/e11_frameB_card.py): s6bcard_loop5js, JOIN_SCHED 384-320-384, the j3cut
patch of JT.make_bd, the trainer's install_star (compiled from the trainer's source text), k4_15 hook = identity, parameters and dials
from native_parameters.load_s33_arrays (research parent, sha a3efb3a1...).
  build(rank or weights, learned=True)  -> card; learned=False zeroes the learned mean correction at layers 1-15 (the lead's 'off')
  wrap(cd, name, fn)                    -> chain fn(layer, value, ctx) AFTER the hook already installed under that name
  truth(rank)                           -> organizers' 1e9 post means (16, 1024) float64
  mcall(rank), planes(rank), pu(rank)   -> 32M all-layer mean/cov truth, 8M third/fourth-order planes, 32M per-unit v/k3/k4 (lazy npz)
Import sets torch.no_grad globally; threads via set_threads(n)."""
import os, sys, ast, inspect, importlib.util
from pathlib import Path
os.environ['JOIN_SCHED'] = '384-320-384'
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent; H = P2 / 'codex_resume_20260926_v2'
TR = H / 'hill_climb_rrpi_detached_star_v1'
for rel in ('fable_analyst_v1', 'fable_covhunter_v1', 'fable_engineer_v1', 'fable_physicist_v1/refit6b', 'fable_porter_v1'):
    sys.path.insert(0, str(P2 / rel))
sys.path.insert(0, str(TR))
import numpy as np, torch
from native_parameters import load_s33_arrays
arrays, DIALS = load_s33_arrays(np)
T = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in arrays.items()}
p = {key[2:]: value for key, value in T.items() if key.startswith('p.')}
e = {key[2:]: value for key, value in T.items() if key.startswith('e.')}
spec = importlib.util.spec_from_file_location('physicist_current_card', P2 / 'fable_physicist_v1/refit6b/s6bcard_loop5js.py')
LC = importlib.util.module_from_spec(spec); spec.loader.exec_module(LC)
assert LC.SCHED == '384-320-384'
bd_source = inspect.getsource(LC.JT.make_bd); assert bd_source.count('for j in range(4):') == 1
exec(compile(ast.parse(bd_source.replace('for j in range(4):', 'for j in range(3):')), str(__file__) + ':j3cut', 'exec'), LC.JT.__dict__)
_nodes = [n for n in ast.parse((TR / 'train_segment.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'install_star']
assert len(_nodes) == 1
_ns = dict(torch=torch); exec(compile(ast.Module(body=_nodes, type_ignores=[]), str(TR / 'train_segment.py') + ':install_star', 'exec'), _ns)
torch.set_grad_enabled(False)
N = 1024


def set_threads(n):
    torch.set_num_threads(int(n))


def weights(rank):
    return LC.JT.weights(rank)


def truth(rank):
    return np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)


def mcall(rank):
    return np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')


def planes(rank, tag=''):
    return np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}{tag}.npz')


def pu(rank, tag='P32M'):
    return np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_{tag}.npz')


def build(w, learned=True, dials=None):
    """w: rank (int) or weights array. dials: optional dict overriding DIALS entries (floats)."""
    if isinstance(w, (int, np.integer)): w = weights(int(w))
    D = dict(DIALS)
    if dials: D.update(dials)
    LC.JT.DIALS_T = {k: torch.tensor(v, dtype=torch.float64) for k, v in D.items()}
    cd = LC.card(torch, w, p, T['a'], T['c'], e, fine_ckpt=False)
    cd.hooks['k4_15'] = lambda layer, value, ctx: value
    _ns['install_star'](cd)
    if not learned:
        cd.hooks['delta'] = lambda layer, d, ctx: d if layer == 0 else torch.zeros_like(d)
    return cd


def wrap(cd, name, fn):
    old = cd.hooks.get(name)
    if old is None: cd.hooks[name] = fn
    else: cd.hooks[name] = lambda layer, value, ctx: fn(layer, old(layer, value, ctx), ctx)
    return cd


def run(cd):
    return cd.run(False).numpy().astype(np.float64)


def mse(pred, tp):
    return ((pred - tp) ** 2).mean(1)
