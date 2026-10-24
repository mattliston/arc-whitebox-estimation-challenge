#!/usr/bin/env python3
"""TAG fable-physicist (RF6-1): joint refit of the learned closure on S6b physics (analyst's s6bcard.py: S2 card + GN knobs v3 + L15 row cut
+ the epoch-2 refit literals; torch, CPU). Derived from fable_analyst_v1/refit_s2.py (same loss, batches, cosine schedule, fine checkpoints).
Groups: 'full' = GNN mean-hook head/head_b + birth coefficients A, cstar + all covariance-edge parameters (3,437); 'full+P' adds the GNN
trunk (the network weights msg/enc0/enc2/enc4, 44,144) as a SECOND optimizer group with its own lr (--lr-trunk) and warmup (--warmup-trunk).
Loss = final-layer MSE vs truthpost; Adam; lr = base * cosine(step/steps) * min(1, (step+1)/warmup) per group; grad-norm clip --clip over
all trained parameters. VALID evaluated before and after every epoch; parameters saved per epoch. A SPIKE line is printed when an update's
batch loss exceeds 1.3x the pre-refit VALID mean (the drop9-style blow-up; the lead's remedy is --clip 0.5).
--eval-only --resume X --nvalid 40 evaluates saved parameters (or the card's own literals without --resume) on the first N valid nets.
usage: refit_s6b.py --tag T --groups full|full+P [--lr 1e-4] [--lr-trunk 5e-5] [--warmup 3] [--warmup-trunk 5] [--clip 1.0] [--epochs 2]
       [--ntrain 160] [--nvalid 16] [--batch 4] [--threads 5] [--resume params.pt] [--card s6bcard] [--eval-only]"""
import sys, json, math, time, argparse, importlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]; AN = P2 / 'fable_analyst_v1'; sys.path.insert(0, str(AN))
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--groups', default='full'); ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--lr-trunk', type=float, default=5e-5); ap.add_argument('--warmup', type=int, default=3); ap.add_argument('--warmup-trunk', type=int, default=5)
ap.add_argument('--clip', type=float, default=1.0); ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--ntrain', type=int, default=160)
ap.add_argument('--nvalid', type=int, default=16); ap.add_argument('--batch', type=int, default=4); ap.add_argument('--threads', type=int, default=5)
ap.add_argument('--resume', default=None); ap.add_argument('--card', default='s6bcard'); ap.add_argument('--eval-only', action='store_true')
args = ap.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
cardmod = importlib.import_module(args.card); model = cardmod.model
OUT = HERE / args.tag; OUT.mkdir(exist_ok=True)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
TRAIN, VALID = inputs['train'][:args.ntrain], inputs['valid'][:args.nvalid]
p, a, c, e = (cardmod.payload(torch) if hasattr(cardmod, 'payload') else model.torch_payload(torch, np, False))   # s6bcard.payload(torch, params=...)
if args.resume:
    sv = torch.load(args.resume)
    with torch.no_grad():
        for k in p: p[k].copy_(sv['p'][k])
        a.copy_(sv['a']); c.copy_(sv['c'])
        for k in e: e[k].copy_(sv['e'][k])
full = [p['head'], p['head_b'], e['output'], e['bias'], a, c] + [e[k] for k in e if k not in ('output', 'bias')]
# trunk = the GNN's network weights only (covhunter's set, 44,144 params). NOT centers/scales/rms/gain/first: those are feature
# normalization constants and a flag; rms (~5.6e-5) moved ~18% by Adam's first sign step and blew up the first (B) run.
TRUNK_KEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')
trunk = [p[k] for k in TRUNK_KEYS] if args.groups == 'full+P' else []
assert args.groups in ('full', 'full+P')
for x in full + trunk: x.requires_grad_(not args.eval_only)
def weights(r): return np.load(STORE / Path(man[r]['path']).parent / 'weights.npy')
def truth(r): return torch.tensor(np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64))
def loss(rank, grad):
    card = cardmod.card(torch, weights(rank), p, a, c, e, fine_ckpt=grad)
    if grad: pred = card.run(True)
    else:
        with torch.no_grad(): pred = card.run(False)
    return ((pred[15].to(torch.float64) - truth(rank)) ** 2).mean()
def evaluate(label):
    L = {r: float(loss(r, False)) for r in VALID}
    print(json.dumps(dict(eval=label, mean=float(np.mean(list(L.values()))), n=len(L))), flush=True); return L
if args.eval_only:
    L = evaluate(f'eval_only_{Path(args.resume).stem if args.resume else "card_literals"}_n{len(VALID)}')
    json.dump(dict(args=vars(args), valid=L), open(OUT / f'eval_n{len(VALID)}.json', 'w'), indent=1); sys.exit(0)
log = dict(args=vars(args), epochs=[], n_params=dict(full=int(sum(x.numel() for x in full)), trunk=int(sum(x.numel() for x in trunk))))
log['valid_before'] = evaluate('valid_before'); base = float(np.mean(list(log['valid_before'].values())))
print(json.dumps(dict(tag=args.tag, groups=args.groups, n_params=log['n_params'], lr=args.lr, lr_trunk=args.lr_trunk if trunk else None, clip=args.clip)), flush=True)
json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
groups = [dict(params=full, lr=args.lr, base=args.lr, warm=args.warmup)]
if trunk: groups.append(dict(params=trunk, lr=args.lr_trunk, base=args.lr_trunk, warm=args.warmup_trunk))
opt = torch.optim.Adam([dict(params=g['params'], lr=g['lr']) for g in groups]); steps = args.epochs * math.ceil(len(TRAIN) / args.batch); step = 0
for ep in range(args.epochs):
    order = rng.permutation(TRAIN).tolist(); tl = []; t0 = time.time()
    for i in range(0, len(order), args.batch):
        for pg, g in zip(opt.param_groups, groups):
            pg['lr'] = g['base'] * 0.5 * (1 + math.cos(math.pi * step / steps)) * min(1.0, (step + 1) / g['warm'])
        opt.zero_grad(set_to_none=True); bl = []
        for r in order[i:i + args.batch]:
            L = loss(r, True) * 1e8 / args.batch; L.backward(); bl.append(float(L) * args.batch / 1e8)
        tl += bl
        gn = float(torch.nn.utils.clip_grad_norm_(full + trunk, args.clip)); opt.step(); step += 1
        rec = dict(epoch=ep, update=step, batch_mean=float(np.mean(bl)), train_recent=float(np.mean(tl[-8:])), grad_norm=round(gn, 4),
                   lr=[round(pg['lr'], 9) for pg in opt.param_groups], elapsed=round(time.time() - t0))
        print(json.dumps(rec), flush=True)
        if np.mean(bl) > 1.3 * base: print(json.dumps(dict(SPIKE=True, update=step, batch_mean=float(np.mean(bl)), valid_before=base)), flush=True)
    v = evaluate(f'valid_epoch{ep}'); log['epochs'].append(dict(epoch=ep, train=float(np.mean(tl)), valid=v)); json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
    torch.save(dict(p={k: t.detach().clone() for k, t in p.items()}, a=a.detach().clone(), c=c.detach().clone(), e={k: t.detach().clone() for k, t in e.items()}), OUT / f'params_epoch{ep}.pt')
print(json.dumps(dict(done=True)), flush=True)
