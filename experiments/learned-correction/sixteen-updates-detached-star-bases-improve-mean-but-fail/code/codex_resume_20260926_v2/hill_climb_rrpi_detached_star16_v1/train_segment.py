"""Fixed detached-star continuation: resume original update8 through update16."""
from pathlib import Path
import ast
import ctypes
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time

HERE = Path(__file__).resolve().parent
H = HERE.parent
P2 = H.parent
RANKS = json.loads((HERE / 'data_roster.json').read_text())['training_ranks']
TRUNK = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')
DIAL_LR = dict(X1=.002, X0=.0032, X2=.0032, bscale=.0008,
               s_carry=.0002, s_new=.0008, c0=.002, k4a=.004, k4b=.004, k4c=.004)

def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)

def write(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')

protocol_path = HERE / 'protocol.json'
assert len(sys.argv) == 3 and pin(protocol_path)['sha256'] == sys.argv[1]
SEGMENT = int(sys.argv[2])
assert 2 <= SEGMENT < 4
START, STOP = 4 * SEGMENT, 4 * (SEGMENT + 1)
protocol = json.loads(protocol_path.read_text())
assert protocol['root_admitted'] is True and protocol['training_ranks'] == RANKS
assert protocol['updates'] == 16 and protocol['batch'] == 4 and protocol['schedule_steps'] == 120
assert protocol['continuation']['segment_start'] == 2 and protocol['continuation']['segment_stop'] == 4
for record in protocol['pins']:
    assert pin(record['path']) == record, record['path']
assert len(protocol['data']) == 160
assert [item['rank'] for item in protocol['data']] == RANKS
assert all(os.environ.get(k) == '1' for k in
           ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
OUT = HERE / ('segment_%02d' % SEGMENT)
OUT.mkdir()
CKPT = HERE / ('resume_after_%03d.pt' % STOP)
assert not CKPT.exists()
os.environ['JOIN_SCHED'] = '384-320-384'
for rel in ('fable_analyst_v1', 'fable_covhunter_v1', 'fable_engineer_v1',
            'fable_physicist_v1/refit6b', 'fable_porter_v1'):
    sys.path.insert(0, str(P2 / rel))
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from native_parameters import load_s33_arrays, FROZEN_P, SOURCE, SOURCE_SHA

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
parts = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
assert len(RANKS) == len(set(RANKS)) == 160
assert RANKS == parts['train'][:160]
assert not set(RANKS) & set(parts['valid'])
rng = np.random.default_rng(0)
ORDER = rng.permutation(RANKS).tolist() + rng.permutation(RANKS).tolist()
BATCHES = [ORDER[i:i+4] for i in range(0, 320, 4)]
DATA = {item['rank']:item for item in protocol['data']}
for rank in {r for b in BATCHES[START:STOP] for r in b}:
    for key in ('weights', 'truth'):
        assert pin(DATA[rank][key]['path']) == DATA[rank][key]
assert pin(SOURCE)['sha256'] == SOURCE_SHA == protocol['baseline']['sha256']
arrays, initial_dials = load_s33_arrays(np)
T = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in arrays.items()}
p = {key[2:]: value for key, value in T.items() if key.startswith('p.')}
e = {key[2:]: value for key, value in T.items() if key.startswith('e.')}
full = [p['head'], p['head_b'], e['output'], e['bias'], T['a'], T['c']]
full += [e[key] for key in e if key not in ('output', 'bias')]
trunk = [p[key] for key in TRUNK]
assert len(e) == 11 and len(full + trunk) == 23 and len(T) == 28
assert sum(x.numel() for x in full) == 3437 and sum(x.numel() for x in trunk) == 44144
for value in full + trunk:
    value.requires_grad_(True)
assert all(not p[key].requires_grad for key in FROZEN_P)
DL = {key: torch.tensor(value, dtype=torch.float64, requires_grad=key in DIAL_LR)
      for key, value in initial_dials.items()}
trained_dials = [DL[key] for key in DIAL_LR]

card_path = P2 / 'fable_physicist_v1/refit6b/s6bcard_loop5js.py'
spec = importlib.util.spec_from_file_location('s33_two_update_card', card_path)
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)
torch.set_num_threads(1)
torch.set_grad_enabled(True)
assert torch.is_grad_enabled() and torch.get_num_threads() == 1
assert LC.SCHED == '384-320-384'
bd_source = inspect.getsource(LC.JT.make_bd)
assert bd_source.count('for j in range(4):') == 1
exec(compile(ast.parse(bd_source.replace('for j in range(4):', 'for j in range(3):')),
             str(__file__) + ':j3cut', 'exec'), LC.JT.__dict__)
relief = ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
relief.restype = ctypes.c_size_t

def collect():
    gc.collect()
    relief(None, 0)

def install_star(cd):
    # Same forward operations; only the basis gradient follows Opus's detached rule.
    def factor(A):
        with torch.no_grad():
            Ad = A.detach()
            cn = (Ad.to(torch.float64) ** 2).sum(0)
            cols = torch.sort(torch.argsort(-cn, stable=True)[:32]).values
            Q = torch.linalg.qr(Ad[:, cols])[0]
            Q = torch.linalg.qr(Ad @ torch.linalg.qr(Ad.T @ Q)[0])[0]
        return Q, Q.T @ A
    def star(w, A):
        wt = w.T
        Q, B = factor(A.to(torch.float32))
        P = wt @ Q
        V = (wt * wt) @ B.T
        result = ((V @ P.T) / 3).to(torch.float64)
        return torch.diagonal_scatter(result, (P * V).sum(1, dtype=torch.float64))
    def diagonal(w, A):
        wt = w.T
        Q, B = factor(A.to(torch.float32))
        pushed = (wt @ Q) @ B
        return ((pushed * wt) * wt).sum(1, dtype=torch.float64)
    cd.env['_flms_star'] = star
    cd.env['_flms_star_diagonal'] = diagonal

def network_backward(item):
    collect()
    weights = np.load(item['weights']['path'], allow_pickle=False)
    assert weights.shape == (16, 1024, 1024) and np.isfinite(weights).all()
    with np.load(item['truth']['path'], allow_pickle=False) as saved:
        target = torch.from_numpy(np.asarray(saved['mc_post'][15], dtype=np.float64).copy())
    assert target.shape == (1024,) and torch.isfinite(target).all()
    LC.JT.DIALS_T = DL
    cd = LC.card(torch, weights, p, T['a'], T['c'], e, fine_ckpt=False)
    cd.hooks['k4_15'] = lambda layer, value, ctx: value
    install_star(cd)
    original_step = cd.step
    def step(layer, state):
        try:
            return original_step(layer, state)
        finally:
            collect()  # Includes checkpoint recomputation/early stop.
    cd.step = step
    started = time.monotonic()
    prediction = cd.run(True)
    assert prediction.shape == (16, 1024) and torch.isfinite(prediction).all()
    loss = ((prediction[15].to(torch.float64) - target) ** 2).mean()
    assert torch.isfinite(loss)
    value = float(loss.detach())
    (loss * (1e8 / 4)).backward()
    elapsed = time.monotonic() - started
    cd.step = original_step
    del prediction, loss, cd, original_step, step, weights, target
    collect()
    return dict(rank=item['rank'], mse=value, seconds=elapsed)

def restore_frozen():
    for key in FROZEN_P:
        assert T['p.' + key].detach().numpy().tobytes() == arrays['p.' + key].tobytes()
    for key in ('X3', 'c1'):
        assert float(DL[key].detach()) == initial_dials[key] and not DL[key].requires_grad

groups = [dict(params=full, base=1e-4, warm=3), dict(params=trunk, base=5e-5, warm=5)]
groups += [dict(params=[DL[key]], base=value, warm=3) for key, value in DIAL_LR.items()]
opt = torch.optim.Adam([dict(params=g['params'], lr=g['base']) for g in groups])

resume_pin = None
if START:
    previous_root = Path(protocol['continuation']['prior_protocol']['path']).parent if START == 8 else HERE
    previous = previous_root / ('resume_after_%03d.pt' % START)
    prior = json.loads((previous_root / ('segment_%02d' % (SEGMENT-1)) / 'completed.json').read_text())
    expected_protocol_sha = protocol['continuation']['prior_protocol']['sha256'] if START == 8 else sys.argv[1]
    if START == 8:
        assert pin(previous) == protocol['continuation']['prior_checkpoint']
        assert pin(previous_root / 'segment_01/completed.json') == protocol['continuation']['prior_segment_completion']
    assert prior['protocol']['sha256'] == expected_protocol_sha
    assert pin(previous) == prior['checkpoint']
    resume_pin = pin(previous)
    sv = torch.load(previous, map_location='cpu', weights_only=True)
    assert sv['update'] == START and sv['protocol_sha'] == expected_protocol_sha
    assert sv['order'] == ORDER and set(sv['parameters']) == set(T)
    with torch.no_grad():
        for key in T:
            T[key].copy_(sv['parameters'][key])
        for key in DL:
            DL[key].copy_(sv['dials'][key])
    opt.load_state_dict(sv['optimizer'])
    def exact_tree(a, b):
        if torch.is_tensor(a):
            return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(exact_tree(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(exact_tree(x,y) for x,y in zip(a,b))
        return type(a) is type(b) and a == b
    assert exact_tree(opt.state_dict(), sv['optimizer']), 'Optimizer state did not resume exactly'
    torch.set_rng_state(sv['torch_rng'])
    # Reload exact parameter and optimizer tensors; no fresh Adam or scheduler reset.
    for key in T:
        assert torch.equal(T[key].detach(), sv['parameters'][key])
    assert all(float(DL[key].detach()) == float(sv['dials'][key]) for key in DL)
    del sv
restore_frozen()
write(OUT / 'manifest.json', dict(protocol=pin(protocol_path), baseline=pin(SOURCE),
      trainer=pin(__file__), ranks=RANKS, updates=16, batch=4, schedule_steps=120, segment=SEGMENT, start=START, stop=STOP,
      torch=torch.__version__, numpy=np.__version__, runtime=sys.executable,
      original_arrays=28, trainable_arrays=23, trainable_dials=list(DIAL_LR),
      exact_rrpi_initial_arrays=True, native_gradient_certified=False,
      validation_targets_accessed=False, final_targets_accessed=False))
records = []
with (OUT / 'steps.jsonl').open('x') as log:
    for update in range(START, STOP):
        for pg, group in zip(opt.param_groups, groups):
            pg['lr'] = group['base'] * .5 * (1 + math.cos(math.pi * update / 120)) * min(1., (update + 1) / group['warm'])
        opt.zero_grad(set_to_none=True)
        rows = []
        for rank in BATCHES[update]:
            record = network_backward(DATA[rank])
            rows.append(record)
            log.write(json.dumps(dict(event='network_backward', update=update + 1, **record), allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(dict(event='network_backward', update=update + 1, **record)), flush=True)
        for value in full + trunk + trained_dials:
            assert value.grad is not None and torch.isfinite(value.grad).all()
        norm = float(torch.nn.utils.clip_grad_norm_(full + trunk, 1., error_if_nonfinite=True))
        dial_norm = float(torch.nn.utils.clip_grad_norm_(trained_dials, 1., error_if_nonfinite=True))
        opt.step()
        restore_frozen()
        assert all(torch.isfinite(value).all() for value in list(T.values()) + list(DL.values()))
        record = dict(event='update', update=update + 1, rows=rows,
                      lr=[pg['lr'] for pg in opt.param_groups], grad_norm=norm,
                      dial_grad_norm=dial_norm, dials={key: float(value.detach()) for key, value in DL.items()})
        records.append(record)
        log.write(json.dumps(record, allow_nan=False) + '\n')
        log.flush()
        print(json.dumps(record, allow_nan=False), flush=True)
        collect()
restore_frozen()
checkpoint = dict(update=STOP, protocol_sha=sys.argv[1], order=ORDER,
                  parameters={key:value.detach().cpu().clone() for key,value in T.items()},
                  dials={key:value.detach().cpu().clone() for key,value in DL.items()},
                  optimizer=opt.state_dict(), torch_rng=torch.get_rng_state())
with CKPT.open('xb') as stream:
    torch.save(checkpoint, stream)
endpoint_pin = dials_pin = None
if STOP == 16:
    endpoint = {key: value.detach().cpu().numpy().copy() for key, value in T.items()}
    with (HERE / 'endpoint.npz').open('xb') as stream:
        np.savez(stream, **endpoint)
    write(HERE / 'endpoint_dials.json', {key: float(value.detach()) for key, value in DL.items()})
    endpoint_pin, dials_pin = pin(HERE / 'endpoint.npz'), pin(HERE / 'endpoint_dials.json')
write(OUT / 'completed.json', dict(completed=True, start=START, optimizer_updates=STOP,
      training_ranks=RANKS, segment_ranks=BATCHES[START:STOP], checkpoint=pin(CKPT),
      resumed_from=resume_pin, endpoint=endpoint_pin, dials=dials_pin,
      protocol=pin(protocol_path), trainer=pin(__file__), baseline=pin(SOURCE),
      endpoint_selected=(STOP == 16), native_gradient_certified=False))
print(json.dumps(dict(completed=True, start=START, optimizer_updates=STOP, final_endpoint=STOP==16)), flush=True)
