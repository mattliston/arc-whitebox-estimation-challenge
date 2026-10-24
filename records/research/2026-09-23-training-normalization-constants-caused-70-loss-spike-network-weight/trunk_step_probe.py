#!/usr/bin/env python3
"""TAG fable-physicist (RF6-2): how large a step can the GNN trunk take on S6b physics? Adam's first step is lr * sign(g) per parameter
whatever the gradient scale (so a global grad clip does not shrink it). Gradient of the batch-1 loss (the refit's first batch: TRAIN order
from default_rng(0)) w.r.t. the trunk (every GNN parameter except head/head_b); then trunk-only steps, other groups fixed:
  sign  : theta - lr * sign(g)                         (Adam's first step)
  grad  : theta - (lr * sqrt(n) / |g|) * g             (plain gradient step of the same length)
scored on batch 2 (held out from the gradient) and batch 1, relative to the unmodified card. Same for the 'full' group as a reference.
usage: trunk_step_probe.py <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]; sys.path.insert(0, str(P2 / 'fable_analyst_v1'))
import numpy as np, torch
import s6bcard
torch.set_num_threads(int(sys.argv[1]))
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
order = np.random.default_rng(0).permutation(inputs['train'][:160]).tolist(); B1, B2 = order[:4], order[4:8]
p, a, c, e = s6bcard.payload(torch)
trunk_keys = [k for k in p if k not in ('head', 'head_b')]
full = [p['head'], p['head_b'], e['output'], e['bias'], a, c] + [e[k] for k in e if k not in ('output', 'bias')]
W = {r: np.load(STORE / Path(man[r]['path']).parent / 'weights.npy') for r in B1 + B2}
T = {r: torch.tensor(np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)) for r in B1 + B2}
def loss(r, grad):
    cd = s6bcard.card(torch, W[r], p, a, c, e, fine_ckpt=grad)
    if grad: pred = cd.run(True)
    else:
        with torch.no_grad(): pred = cd.run(False)
    return ((pred[15].to(torch.float64) - T[r]) ** 2).mean()
def batch(rs): return float(np.mean([float(loss(r, False)) for r in rs]))
t0 = time.time()
for x in [p[k] for k in trunk_keys] + full: x.requires_grad_(True)
for r in B1: (loss(r, True) * 1e8 / 4).backward()
gt = [p[k].grad.detach().clone() for k in trunk_keys]; gf = [x.grad.detach().clone() for x in full]
for x in [p[k] for k in trunk_keys] + full: x.requires_grad_(False); x.grad = None
ref1, ref2 = batch(B1), batch(B2)
nt = sum(g.numel() for g in gt); gnt = float(torch.sqrt(sum((g.double() ** 2).sum() for g in gt)))
nf = sum(g.numel() for g in gf); gnf = float(torch.sqrt(sum((g.double() ** 2).sum() for g in gf)))
print(json.dumps(dict(batch1=B1, batch2=B2, ref_b1=ref1, ref_b2=ref2, trunk_params=nt, trunk_grad_norm=gnt, full_params=nf, full_grad_norm=gnf,
                      wall=round(time.time() - t0, 1))), flush=True)
def probe(tensors, grads, n, gn, lrs, label):
    orig = [x.detach().clone() for x in tensors]
    for kind in ('sign', 'grad'):
        for lr in lrs:
            with torch.no_grad():
                for x, o, g in zip(tensors, orig, grads):
                    x.copy_(o - (lr * torch.sign(g) if kind == 'sign' else (lr * n ** 0.5 / gn) * g).to(o.dtype))
            b1, b2 = batch(B1), batch(B2)
            print(json.dumps(dict(group=label, kind=kind, lr=lr, b1=round(b1 / ref1 - 1, 5), b2=round(b2 / ref2 - 1, 5))), flush=True)
        with torch.no_grad():
            for x, o in zip(tensors, orig): x.copy_(o)
probe([p[k] for k in trunk_keys], gt, nt, gnt, (1e-7, 3e-7, 1e-6, 3e-6, 1e-5), 'trunk')
probe(full, gf, nf, gnf, (1e-5, 3.3e-5, 1e-4), 'full')
print(json.dumps(dict(done=True, wall=round(time.time() - t0, 1))), flush=True)
