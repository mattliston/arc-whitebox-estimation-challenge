"""Bounded target-free response comparison of an existing Torch proposal card.

This is not a matched native forward or full-Jacobian certificate. No fit.
Run only under the root-owned guard, once per prescribed TRAIN rank.
"""
from pathlib import Path
import ast
import ctypes
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
P2 = H2.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol_path = HERE / 'protocol.json'
protocol = json.loads(protocol_path.read_text())
assert sha(protocol_path) == sys.argv[2]
assert protocol['root_admitted'] is True and protocol['fitting_admitted'] is False
rank = int(sys.argv[1])
assert rank in (35, 100)
for pin in protocol['pins']:
    assert sha(pin['path']) == pin['sha256'], pin['path']
out = HERE / ('torch_rank' + str(rank))
out.mkdir()
os.environ['JOIN_SCHED'] = '384-320-384'
for rel in ('fable_analyst_v1', 'fable_covhunter_v1', 'fable_engineer_v1',
            'fable_physicist_v1/refit6b', 'fable_porter_v1'):
    sys.path.insert(0, str(P2 / rel))
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from torch.autograd import forward_ad as fad
from direction_common import make_directions, perturb
from native_parameters import load_s33_arrays

card_path = P2 / 'fable_physicist_v1/refit6b/s6bcard_loop5js.py'
spec = importlib.util.spec_from_file_location('s33_response_card', card_path)
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
assert LC.SCHED == '384-320-384'

# Small declared semantic patches only: removed j3, S32's 32-column star,
# and unscaled terminal k4. Old-core/tail/common-factor/precision gaps remain.
bd_source = inspect.getsource(LC.JT.make_bd)
assert bd_source.count('for j in range(4):') == 1
bd_patched = bd_source.replace('for j in range(4):', 'for j in range(3):')
exec(compile(ast.parse(bd_patched), str(HERE / 'torch_probe.py') + ':j3cut', 'exec'), LC.JT.__dict__)

libsystem = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
relief = libsystem.malloc_zone_pressure_relief
relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
relief.restype = ctypes.c_size_t
def collect():
    gc.collect()
    relief(None, 0)

arrays, dials = load_s33_arrays(np)
directions, dial_directions, direction_record = make_directions(np, arrays, dials)
store = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((store / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights_path = store / Path(man[rank]['path']).parent / 'weights.npy'
weights = np.load(weights_path, allow_pickle=False)
parts = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
assert rank in parts['train'] and rank not in parts['valid']

def digest(x):
    x = fad.unpack_dual(x).primal
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def install_star(cd, record):
    def factor(A):
        cn = (A.to(torch.float64) ** 2).sum(0)
        cols = torch.sort(torch.argsort(-cn, stable=True)[:32]).values
        Q = torch.linalg.qr(A[:, cols])[0]
        Q = torch.linalg.qr(A @ torch.linalg.qr(A.T @ Q)[0])[0]
        record['star_columns'].append(digest(cols))
        return Q, Q.T @ A
    def star(w, A):
        wt = w.T
        Q, B = factor(A.to(torch.float32))
        P = wt @ Q
        T = (wt * wt) @ B.T
        result = ((T @ P.T) / 3).to(torch.float64)
        return torch.diagonal_scatter(result, (P * T).sum(1, dtype=torch.float64))
    def diagonal(w, A):
        wt = w.T
        Q, B = factor(A.to(torch.float32))
        pushed = (wt @ Q) @ B
        return ((pushed * wt) * wt).sum(1, dtype=torch.float64)
    cd.env['_flms_star'] = star
    cd.env['_flms_star_diagonal'] = diagonal

def run_case(name, epsilon, dual):
    collect()
    start = time.monotonic()
    aa, dd = perturb(np, arrays, dials, directions, dial_directions, epsilon)
    record = {'name': name, 'epsilon': epsilon, 'dual': dual, 'rows': [], 'star_columns': []}
    with torch.no_grad(), fad.dual_level():
        tt = {}
        for key, value in aa.items():
            tensor = torch.from_numpy(value.copy())
            if dual:
                tensor = fad.make_dual(tensor, torch.from_numpy(directions[key].astype(np.float32)))
            tt[key] = tensor
        p = {key[2:]: value for key, value in tt.items() if key.startswith('p.')}
        e = {key[2:]: value for key, value in tt.items() if key.startswith('e.')}
        D = {}
        for key, value in dd.items():
            v = torch.tensor(value, dtype=torch.float64)
            D[key] = fad.make_dual(v, torch.tensor(dial_directions[key], dtype=torch.float64)) if dual else v
        LC.JT.DIALS_T = D
        cd = LC.card(torch, weights, p, tt['a'], tt['c'], e, fine_ckpt=False)
        cd.hooks['k4_15'] = lambda layer, value, ctx: value
        install_star(cd, record)
        old_select, old_floor = cd.select, cd.select_floor
        def note(result, kind):
            record['rows'].append({'layer': cd.call_layer, 'kind': kind, 'ids': digest(result[0])})
            return result
        cd.select = lambda score: note(old_select(score), 'select')
        cd.select_floor = lambda score, floor: note(old_floor(score, floor), 'select_floor')
        state = cd.initial()
        values, tangents = [], []
        for layer in range(16):
            state = cd.step(layer, state)
            primal, tangent = fad.unpack_dual(state[0].clamp_min(0))
            values.append(primal.detach().cpu().to(torch.float64).numpy().copy())
            tangents.append(np.zeros_like(values[-1]) if tangent is None else
                            tangent.detach().cpu().to(torch.float64).numpy().copy())
            collect()
        values, tangents = np.stack(values), np.stack(tangents)
        assert values.shape == (16, 1024) and np.isfinite(values).all() and np.isfinite(tangents).all()
    with (out / (name + '.npz')).open('xb') as fp:
        np.savez(fp, prediction=values, tangent=tangents)
    record.update(elapsed_s=time.monotonic() - start, finite=True,
                  prediction_sha256=hashlib.sha256(values.tobytes()).hexdigest())
    with (out / (name + '.json')).open('x') as fp:
        json.dump(record, fp, indent=2)
    print(json.dumps({'event': 'completed', 'rank': rank, 'case': name, 'seconds': record['elapsed_s']}), flush=True)
    del cd, state, tt, p, e, D, values, tangents
    collect()

with (out / 'manifest.json').open('x') as fp:
    json.dump({'protocol_sha256': sha(protocol_path), 'rank': rank,
               'direction': direction_record, 'weights_sha256': sha(weights_path),
               'runtime': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__,
               'targets_accessed': False, 'optimizer_steps': 0,
               'full_native_gradient_certification': False}, fp, indent=2)
for name, eps, dual in [('zero', 0., True), ('plus_h', .01, False), ('minus_h', -.01, False),
                        ('plus_half', .005, False), ('minus_half', -.005, False)]:
    run_case(name, eps, dual)
with (out / 'completion.json').open('x') as fp:
    json.dump({'completed': True, 'rank': rank, 'cases': 5, 'targets_accessed': False,
               'optimizer_steps': 0}, fp)
