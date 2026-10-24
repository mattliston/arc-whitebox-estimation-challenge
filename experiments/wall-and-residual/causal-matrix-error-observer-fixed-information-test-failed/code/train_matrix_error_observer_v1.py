"""Locked20/4,100epoch source fit; actual frozen-bank final mean objective."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from matrix_error_observer_data_v1 import load
from matrix_error_observer_v1 import analytic_basis
from matrix_error_observer_torch_v1 import predict

HERE = Path(__file__).resolve().parent
G22 = HERE.parent/'public_g22_operator_features_v1'
OUT = HERE/'matrix_error_observer_training_verdict_v1.json'
assert not OUT.exists()
for name in ('small_certificate', 'wide_certificate', 'rows'):
    assert json.loads((HERE/f'matrix_error_observer_{name}_v1.json').read_text())['passed']
assert torch.backends.mps.is_available()
torch.set_num_threads(2)
torch.manual_seed(930072)
valid = [37, 39, 42, 48]
train = [r for r in range(35, 59) if r not in valid]
started = time.time()
cpu, banks, truths = {}, {}, {}
sum2, count = np.zeros(14, np.float64), 0
for r in range(35, 59):
    d, b = load(r)
    cpu[r], banks[r] = d, b
    if r in train:
        f = analytic_basis(d['x'])[1:15].astype(np.float64)
        sum2 += (f*f).sum(axis=(0, 1))
        count += f.shape[0]*f.shape[1]
scale = np.maximum(np.sqrt(sum2/count), 1e-6).astype(np.float32)
# All inputs and TRAIN-only normalization fixed before targets are read.
for r in range(35, 59):
    truths[r] = np.load(G22/'truthpost'/f'truthpost_rank{r:04d}.npz')['mc_post'][-1].astype(np.float64)
norm = float(np.mean([np.mean((truths[r]-banks[r])**2) for r in train]))
dev = torch.device('mps')
tscale = torch.tensor(scale, device=dev)
data, targets, clips = {}, {}, {}
for r in range(35, 59):
    data[r] = {k: torch.tensor(np.array(v), device=dev) for k, v in cpu[r].items()}
    targets[r] = torch.tensor((truths[r]-banks[r]).astype(np.float32), device=dev)
    clips[r] = torch.tensor(-banks[r].astype(np.float32), device=dev)
del cpu
torch.mps.synchronize()
print(json.dumps(dict(stage='ready', wall_s=time.time()-started, train_bank_mse=norm,
                      mps_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())), flush=True)
fit_start = time.time()
arms = {}
timeout = False
for memory in (True, False):
    name = 'memory' if memory else 'local'
    theta = torch.nn.Parameter(torch.zeros((3, 14, 14), device=dev))
    optimizer = torch.optim.Adam([theta], lr=.01)
    rng = np.random.default_rng(930072)
    history = []
    for epoch in range(100):
        lr = .001+.009*(1+np.cos(np.pi*epoch/99))/2
        optimizer.param_groups[0]['lr'] = lr
        losses = []
        for r in rng.permutation(train):
            if time.time()-fit_start > 600:
                timeout = True
                break
            optimizer.zero_grad(set_to_none=True)
            correction = predict(data[int(r)], theta, tscale, memory)
            applied = torch.maximum(correction, clips[int(r)])
            mse = torch.mean((applied-targets[int(r)])**2)
            loss = mse/norm+1e-3*torch.mean(theta*theta)
            if not torch.isfinite(loss).item():
                raise FloatingPointError((name, epoch, int(r)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], 5.)
            optimizer.step()
            losses.append(float(mse.detach().cpu()))
        torch.mps.synchronize()
        record = dict(epoch=epoch+1, lr=lr, train_online_mse=float(np.mean(losses)) if losses else None,
                      fit_wall_s=time.time()-fit_start)
        history.append(record)
        print(json.dumps(dict(arm=name, **record)), flush=True)
        if timeout:
            break
    if timeout:
        arms[name] = dict(completed=False, history=history)
        break
    coefficients = theta.detach().cpu().numpy()
    np.savez(HERE/f'matrix_error_observer_{name}_model_v1.npz', theta=coefficients, feature_scale=scale)
    rows = []
    corrections = {}
    with torch.no_grad():
        for r in valid:
            correction = predict(data[r], theta, tscale, memory).cpu().numpy()
            estimate = np.maximum(banks[r]+correction.astype(np.float64), 0)
            raw = float(np.mean((estimate-truths[r])**2))
            bank = float(np.mean((banks[r]-truths[r])**2))
            rows.append(dict(rank=r, bank_mse=bank, observer_mse=raw, ratio=raw/bank))
            corrections[f'rank_{r:04d}'] = correction
    np.savez(HERE/f'matrix_error_observer_{name}_held_corrections_v1.npz', **corrections)
    arms[name] = dict(completed=True, history=history, held_out=rows,
                      pooled_mse=float(np.mean([row['observer_mse'] for row in rows])))
    print(json.dumps(dict(arm=name, held_out=rows, pooled_mse=arms[name]['pooled_mse'])), flush=True)
passed = False
gates = {}
if not timeout:
    bank = float(np.mean([r['bank_mse'] for r in arms['memory']['held_out']]))
    ratio = arms['memory']['pooled_mse']/bank
    vs_local = arms['memory']['pooled_mse']/arms['local']['pooled_mse']
    wins = sum(r['ratio'] < 1 for r in arms['memory']['held_out'])
    gates = dict(memory_bank_ratio=ratio, memory_local_ratio=vs_local, bank_wins=wins,
                 required_bank_ratio=.75, required_local_ratio=.85, required_wins=3)
    passed = ratio <= .75 and vs_local <= .85 and wins >= 3
result = dict(passed=passed, timeout=timeout, train_ranks=train, held_out_ranks=valid,
              train_bank_mse=norm, seed=930072, epochs=100, arms=arms, gates=gates,
              fit_wall_s=time.time()-fit_start, total_wall_s=time.time()-started,
              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              source_hashes={s: hashlib.sha256((HERE/s).read_bytes()).hexdigest() for s in
                             ('matrix_error_observer_v1.py', 'matrix_error_observer_torch_v1.py',
                              'matrix_error_observer_data_v1.py', 'train_matrix_error_observer_v1.py')},
              qualification='Fixed-source auxiliary error observer; main P/Q card unchanged. Existing1e9 final means,20train/4held. No covariance labels, new MC or adjusted/native score.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'arms'}, indent=2), flush=True)
