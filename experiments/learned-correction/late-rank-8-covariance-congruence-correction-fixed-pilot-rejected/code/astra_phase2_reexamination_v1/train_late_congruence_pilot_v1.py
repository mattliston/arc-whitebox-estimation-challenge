"""Predeclared rank8 PSD covariance write; five epochs, final epoch only."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from late_congruence_training_shadow_v1 import TrainCongruenceShadow

here = Path(__file__).resolve().parent
out = here/'late_congruence_training_pilot_v1.json'
payload = here/'payload_late_congruence_pilot_v1.npz'
action = here/'action_late_congruence_pilot_v1.npz'
progress = here/'late_congruence_pilot_progress_v1'
assert all(not p.exists() for p in (out, payload, action, progress))
assert json.loads((here/'late_congruence_fidelity_v1.json').read_text())['fidelity_pass']
VALID = (37, 39, 42, 48)
TRAIN = tuple(r for r in range(35, 59) if r not in VALID)
torch.manual_seed(930057)
torch.set_num_threads(2)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
Pnp = dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))
frozen = {'centers', 'scales', 'rms', 'first'}
parameters = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(v, device=dev, dtype=torch.float32))
                                    for k, v in Pnp.items() if k not in frozen})

# Normalization and action initialization use baseline TRAIN tapes only.
centers, scales = np.zeros((16, 64), np.float32), np.ones((16, 64), np.float32)
for l in range(10, 15):
    h = np.concatenate([np.load(here/'rows_current_transport_shadow_v2'/f'row_{r:04d}'/'state_prev.npy')[l+1]
                        for r in TRAIN], axis=0).astype(np.float64)
    centers[l] = h.mean(0)
    scales[l] = np.maximum(h.std(0), .01)
rng = np.random.default_rng(930057)
v0 = np.zeros((65, 8), np.float32)
v0[64, 0] = 1.
v0[:64, 1:] = np.linalg.qr(rng.standard_normal((64, 7)), mode='reduced')[0]
covparams = torch.nn.ParameterDict({
    'u': torch.nn.Parameter(torch.zeros((5, 65, 8), device=dev)),
    'v': torch.nn.Parameter(torch.tensor(v0, device=dev))})
Qfixed = {k: torch.tensor(v, device=dev) for k, v in dict(centers=centers, scales=scales).items()}
progress.mkdir()
# Training-target reads start here, after the written fixed pilot plan.
labels = {r: torch.tensor(np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{r:04d}.npz')['mc_post'],
                           device=dev, dtype=torch.float32) for r in TRAIN+VALID}
base_cpu = {r: np.load(here/'rows_current_transport_shadow_v2'/f'row_{r:04d}'/'baseline.npy') for r in TRAIN+VALID}
normalizer = float(np.mean([np.mean((np.maximum(base_cpu[r][-1], 0)-labels[r][-1].cpu().numpy())**2) for r in TRAIN]))
opt = torch.optim.Adam([dict(params=parameters.parameters(), lr=5e-5),
                        dict(params=covparams.parameters(), lr=3e-3)])
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: .1+.9*(1+np.cos(np.pi*step/100))/2)
weights = torch.tensor([1., 1., 1., 1., 1., 4.], device=dev)


def run(rank, training):
    s = TrainCongruenceShadow(rank, dev)
    P = dict(s.P0)
    P.update(dict(parameters.items()))
    Q = dict(Qfixed)
    Q.update(dict(covparams.items()))
    dm = torch.zeros(1024, device=dev)
    dc = torch.zeros((1024, 1024), device=dev)
    dh = torch.zeros((1024, 64), device=dev)
    parts = []
    for l in range(10, 16):
        def block(dm, dc, dh, l=l):
            return s.step_train(l, dm, dc, dh, P, Q)
        if training:
            dm, dc, dh = checkpoint(block, dm, dc, dh, use_reentrant=False)
        else:
            dm, dc, dh = block(dm, dc, dh)
        parts.append(dm)
    pred = (s.data['baseline'][10:]+torch.stack(parts)).clamp_min(0)
    mse = ((pred-labels[rank][10:])**2).mean(1)
    loss = (mse*weights).sum()/weights.sum()/normalizer
    return loss, mse, s


def evaluate():
    rows = []
    with torch.no_grad():
        for rank in VALID:
            loss, mse, s = run(rank, False)
            old = float(np.mean((np.maximum(base_cpu[rank][-1], 0)-labels[rank][-1].cpu().numpy())**2))
            val = float(mse[-1].cpu())
            rows.append(dict(rank=rank, shadow_mse=val, parent_mse=old, ratio=val/old))
            del loss, mse, s
    return rows


started = time.time()
initial = evaluate()
assert all(abs(r['ratio']-1) < .002 for r in initial), 'parent initialization mismatch'
history = []
print(json.dumps(dict(device=dev, train=TRAIN, valid=VALID, epochs=5, normalizer=normalizer, initial=initial)), flush=True)
for epoch in range(5):
    losses = []
    for rank in np.random.default_rng(930057+epoch).permutation(TRAIN):
        assert time.time()-started < 900, '15-minute pilot limit'
        loss, mse, s = run(int(rank), True)
        assert torch.isfinite(loss).item(), (epoch, int(rank), 'loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(parameters.parameters())+list(covparams.parameters()), 5.)
        assert torch.isfinite(norm).item(), (epoch, int(rank), 'gradient')
        opt.step()
        sched.step()
        losses.append(float(loss.detach().cpu()))
        del loss, mse, s
    row = dict(epoch=epoch+1, train_normalized_loss=float(np.mean(losses)), valid=evaluate(), wall_s=time.time()-started)
    history.append(row)
    (progress/f'epoch_{epoch+1:02d}.json').write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)
Pout = {k: v.copy() for k, v in Pnp.items()}
Pout.update({k: v.detach().cpu().numpy() for k, v in parameters.items()})
np.savez(payload, **Pout)
Qout = dict(centers=centers, scales=scales)
Qout.update({k: v.detach().cpu().numpy() for k, v in covparams.items()})
np.savez(action, **Qout)
receipt = dict(train=TRAIN, valid=VALID, epochs=5, initial=initial, history=history, wall_s=time.time()-started,
               qualification='Covariance-action shadow training only; no actual-carrier or adjusted gain established.')
out.write_text(json.dumps(receipt, indent=2)+'\n')
print('LATE_CONGRUENCE_PILOT_TRAINED', flush=True)
