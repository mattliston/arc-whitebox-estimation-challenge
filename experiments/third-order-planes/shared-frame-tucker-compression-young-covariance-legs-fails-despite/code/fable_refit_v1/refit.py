#!/usr/bin/env python3
"""TAG fable-refit (RF-2): JOINT REFIT of the 331824 card's learned/closure parameters on the research torch card after a
STRUCTURAL change: --drop-age A zeroes the signed cube-column weights of every cohort aged >= A (A >= 6; 0 = none) — the
accuracy proxy for dropping those cohorts from the old tier (the native bill then falls ~13% at A=9, ~24% at A=7, ~35% at A=6).
Trainable groups (--groups): P (GNN mean hook, 45,200), A (bdnh 64x4), c (bd coefficients 15x4), E (covariance edge + message
context). Loss = final-layer MSE vs the stored truths; Adam, batch of --batch nets per update, cosine lr; VALID evaluated (forward
only) after every epoch; parameters saved per epoch.  usage: refit.py --tag T --drop-age 9 --epochs 3 --lr 3e-4 --batch 4 --threads 8
[--groups P,A,c,E] [--train N] [--valid N] [--eval-only]"""
import sys, os, time, json, math, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--drop-age', type=int, default=0); ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--batch', type=int, default=4); ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--groups', default='P,A,c,E'); ap.add_argument('--train', type=int, default=160); ap.add_argument('--valid', type=int, default=40)
ap.add_argument('--eval-only', action='store_true'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--resume', default=None); ap.add_argument('--eval-ranks', default=None); ap.add_argument('--tier-rank', type=int, default=384); ap.add_argument('--warmup', type=int, default=0); ap.add_argument('--zero-learned', action='store_true'); ap.add_argument('--lr-rest', type=float, default=None)
args = ap.parse_args()
torch.set_num_threads(args.threads); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
model.TIER_RANK = args.tier_rank
OUT = HERE / args.tag; OUT.mkdir(exist_ok=True)
P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
inputs = json.loads((P2 / "codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json").read_text())
TRAIN, VALID = inputs['train'][:args.train], inputs['valid'][:args.valid]
if args.eval_ranks: VALID = [int(x) for x in args.eval_ranks.split(',')]
def weights(rank): return np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
def truth(rank): return torch.tensor(np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"][15].astype(np.float64))
p, a, c, e = model.torch_payload(torch, np, False)
if args.resume:
    saved = torch.load(args.resume)
    for k in p: p[k].copy_(saved['p'][k])
    a.copy_(saved['a']); c.copy_(saved['c'])
    for k in e: e[k].copy_(saved['e'][k])
groups = set(args.groups.split(','))
zeroed = []
if args.zero_learned:
    with torch.no_grad():
        for k in ('head', 'head_b'): p[k].zero_(); zeroed.append(p[k])
        for k in ('output', 'bias'): e[k].zero_(); zeroed.append(e[k])
params = []
if 'P' in groups: params += [v.requires_grad_(True) for v in p.values()]
if 'A' in groups: params.append(a.requires_grad_(True))
if 'c' in groups: params.append(c.requires_grad_(True))
if 'E' in groups: params += [v.requires_grad_(True) for v in e.values()]
zid = {id(v) for v in zeroed}
print(json.dumps(dict(tag=args.tag, drop_age=args.drop_age, tier_rank=args.tier_rank, groups=sorted(groups), n_params=int(sum(v.numel() for v in params)), train=len(TRAIN), valid=len(VALID), lr=args.lr, batch=args.batch, epochs=args.epochs)), flush=True)

class DropCard(model.Card):
    def step(self, layer, state):
        lam, stops = state[3], state[12]
        if args.drop_age and lam.numel():
            mask = torch.ones_like(lam); s0 = 0; hit = False
            for b, s1 in enumerate(stops):
                if layer - b >= args.drop_age and s1 > s0:
                    mask[s0:s1] = 0.0; hit = True
                s0 = s1
            if hit: state = state[:3] + (lam * mask,) + state[4:]
        return super().step(layer, state)

def run_net(rank, grad):
    card = DropCard(torch, weights(rank), p, a, c, e)
    if grad:
        pred = card.run(True)
    else:
        with torch.no_grad(): pred = card.run(False)
    return ((pred[15].to(torch.float64) - truth(rank)) ** 2).mean()

def evaluate(ranks, label):
    losses = {}
    for r in ranks:
        losses[r] = float(run_net(r, False))
    gm = math.exp(np.mean([math.log(v) for v in losses.values()]))
    print(json.dumps(dict(eval=label, mean=float(np.mean(list(losses.values()))), geo_mean=gm, n=len(losses))), flush=True)
    return losses

def save(name):
    torch.save(dict(p={k: v.detach().clone() for k, v in p.items()}, a=a.detach().clone(), c=c.detach().clone(), e={k: v.detach().clone() for k, v in e.items()}), OUT / name)

log = dict(args=vars(args), epochs=[])
v0 = evaluate(VALID, 'valid_before'); log['valid_before'] = v0
json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
if args.eval_only: sys.exit(0)
opt = (torch.optim.Adam([dict(params=[v for v in params if id(v) in zid], lr=args.lr), dict(params=[v for v in params if id(v) not in zid], lr=args.lr_rest)])
       if args.lr_rest is not None else torch.optim.Adam(params, lr=args.lr))
base_lrs = [g['lr'] for g in opt.param_groups]
steps_total = args.epochs * math.ceil(len(TRAIN) / args.batch); step = 0
for epoch in range(args.epochs):
    order = rng.permutation(TRAIN).tolist(); t0 = time.time(); train_losses = []
    for i in range(0, len(order), args.batch):
        batch = order[i:i + args.batch]
        fac = 0.5 * (1 + math.cos(math.pi * step / steps_total)) * (min(1.0, (step + 1) / args.warmup) if args.warmup else 1.0); lr = args.lr * fac
        for g, b in zip(opt.param_groups, base_lrs): g['lr'] = b * fac
        opt.zero_grad(set_to_none=True)
        for r in batch:
            loss = run_net(r, True) * 1e8 / len(batch); loss.backward(); train_losses.append(float(loss) * len(batch) / 1e8)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
        if True:
            print(json.dumps(dict(epoch=epoch, update=step, lr=lr, recent_train_mean=float(np.mean(train_losses[-20:])), elapsed=round(time.time() - t0, 1))), flush=True)
    save(f'params_epoch{epoch}.pt')
    v = evaluate(VALID, f'valid_epoch{epoch}')
    log['epochs'].append(dict(epoch=epoch, train_mean=float(np.mean(train_losses)), valid=v, wall=round(time.time() - t0, 1)))
    json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
print(json.dumps(dict(done=True)), flush=True)
