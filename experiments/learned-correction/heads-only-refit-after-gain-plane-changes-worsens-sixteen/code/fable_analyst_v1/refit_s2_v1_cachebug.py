#!/usr/bin/env python3
"""TAG fable-analyst (AN-36): re-adapt the learned closure to S2's physics (s2card.py: K31 row .85/.5 at L1-14 + gain b-fix gamma .1 +
join every3; torch parity with native S2: r35 -0.02%, r100 +0.06%). Trainable groups: 'heads' = GNN mean-hook head/head_b + covariance-edge
output/bias (as fable_refit_v1/refit_physics.py); 'full' adds the birth coefficients (A, cstar) and all edge parameters. Loss = final-layer
MSE vs truthpost; Adam, batches of 4 nets, cosine lr with 3-update warmup, grad-norm clip 1; VALID evaluated before (= the no-retrain
control) and after every epoch; fine-grained checkpoints keep the fwd+bwd peak RSS at ~6.9 GB.
usage: refit_s2.py --tag T [--groups heads|full] [--ntrain 32] [--nvalid 16] [--epochs 2] [--lr 1e-3] [--threads 3] [--resume params.pt]"""
import sys, json, math, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import s2card
model = s2card.model
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--groups', default='heads'); ap.add_argument('--ntrain', type=int, default=32); ap.add_argument('--nvalid', type=int, default=16)
ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--threads', type=int, default=3); ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--resume', default=None)
args = ap.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
OUT = HERE / args.tag; OUT.mkdir(exist_ok=True)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
TRAIN, VALID = inputs['train'][:args.ntrain], inputs['valid'][:args.nvalid]
p, a, c, e = model.torch_payload(torch, np, False)
if args.resume:
    sv = torch.load(args.resume)
    with torch.no_grad():
        for k in p: p[k].copy_(sv['p'][k])
        a.copy_(sv['a']); c.copy_(sv['c'])
        for k in e: e[k].copy_(sv['e'][k])
params = [p['head'], p['head_b'], e['output'], e['bias']]
if args.groups == 'full':
    params += [a, c] + [e[k] for k in e if k not in ('output', 'bias')]
for x in params: x.requires_grad_(True)
W = {}
def weights(r):
    if r not in W: W[r] = np.load(STORE / Path(man[r]['path']).parent / 'weights.npy')
    return W[r]
def truth(r): return torch.tensor(np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64))
def loss(rank, grad):
    card = s2card.card(torch, weights(rank), p, a, c, e, fine_ckpt=grad)
    if grad: pred = card.run(True)
    else:
        with torch.no_grad(): pred = card.run(False)
    return ((pred[15].to(torch.float64) - truth(rank)) ** 2).mean()
def evaluate(label):
    L = {r: float(loss(r, False)) for r in VALID}
    print(json.dumps(dict(eval=label, mean=float(np.mean(list(L.values()))), n=len(L))), flush=True); return L
log = dict(args=vars(args), epochs=[], n_params=int(sum(x.numel() for x in params))); log['valid_before'] = evaluate('valid_before'); json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
opt = torch.optim.Adam(params, lr=args.lr); steps = args.epochs * math.ceil(len(TRAIN) / args.batch); step = 0
for ep in range(args.epochs):
    order = rng.permutation(TRAIN).tolist(); tl = []; t0 = time.time()
    for i in range(0, len(order), args.batch):
        fac = 0.5 * (1 + math.cos(math.pi * step / steps)) * min(1.0, (step + 1) / 3)
        for g in opt.param_groups: g['lr'] = args.lr * fac
        opt.zero_grad(set_to_none=True)
        for r in order[i:i + args.batch]:
            L = loss(r, True) * 1e8 / args.batch; L.backward(); tl.append(float(L) * args.batch / 1e8)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
        print(json.dumps(dict(epoch=ep, update=step, train_recent=float(np.mean(tl[-8:])), elapsed=round(time.time() - t0))), flush=True)
    v = evaluate(f'valid_epoch{ep}'); log['epochs'].append(dict(epoch=ep, train=float(np.mean(tl)), valid=v)); json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
    torch.save(dict(p={k: t.detach().clone() for k, t in p.items()}, a=a.detach().clone(), c=c.detach().clone(), e={k: t.detach().clone() for k, t in e.items()}), OUT / f'params_epoch{ep}.pt')
