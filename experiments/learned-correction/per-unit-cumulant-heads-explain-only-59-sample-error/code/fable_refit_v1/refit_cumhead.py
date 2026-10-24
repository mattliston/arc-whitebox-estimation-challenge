#!/usr/bin/env python3
"""TAG fable-refit (RF-7): NEW CHANNEL — per-unit k3/k4 multiplicative corrections from the causal GNN hidden state (hstate, 1024x64
from the previous layer), zero-initialized, at layers >= 8 and at the final scalar map: k3_j *= 1 + 0.3 tanh(h_j.w3 + b3),
k4_j *= 1 + 0.3 tanh(h_j.w4 + b4) (K22 diagonal follows k4). 130 new values, trained JOINTLY with all learned groups (refit.py
semantics: warmup + cosine Adam, TRAIN160, VALID40 after each epoch).  usage: refit_cumhead.py --tag T --epochs 2 --lr 1e-4 --warmup 20 --threads 6"""
import sys, os, time, json, math, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--warmup', type=int, default=20); ap.add_argument('--batch', type=int, default=4); ap.add_argument('--threads', type=int, default=6)
ap.add_argument('--valid', type=int, default=40); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--amp', type=float, default=0.3)
ap.add_argument('--head-only', action='store_true'); ap.add_argument('--vamp', type=float, default=0.0); ap.add_argument('--from-layer', type=int, default=8); ap.add_argument('--k21amp', type=float, default=0.0)
args = ap.parse_args()
torch.set_num_threads(args.threads); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
OUT = HERE / args.tag; OUT.mkdir(exist_ok=True)
P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
inputs = json.loads((P2 / "codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json").read_text())
TRAIN, VALID = inputs['train'], inputs['valid'][:args.valid]
def weights(rank): return np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
def truth(rank): return torch.tensor(np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"][15].astype(np.float64))
p, a, c, e = model.torch_payload(torch, np, False)
head = dict(w3=torch.zeros(64, dtype=torch.float32, requires_grad=True), b3=torch.zeros((), dtype=torch.float32, requires_grad=True),
            w4=torch.zeros(64, dtype=torch.float32, requires_grad=True), b4=torch.zeros((), dtype=torch.float32, requires_grad=True),
            wv=torch.zeros(64, dtype=torch.float32, requires_grad=True), bv=torch.zeros((), dtype=torch.float32, requires_grad=True),
            w21=torch.zeros(64, dtype=torch.float32, requires_grad=True), b21=torch.zeros((), dtype=torch.float32, requires_grad=True))
VAMP = args.vamp; K21AMP = args.k21amp; FROM = args.from_layer
params = list(head.values())
if not args.head_only:
    params += [v.requires_grad_(True) for v in p.values()] + [a.requires_grad_(True), c.requires_grad_(True)] + [v.requires_grad_(True) for v in e.values()]
print(json.dumps(dict(tag=args.tag, n_params=int(sum(v.numel() for v in params)), head_only=args.head_only, amp=args.amp, lr=args.lr, warmup=args.warmup, epochs=args.epochs)), flush=True)
AMP = args.amp

class CumHeadCard(model.Card):
    """Injects per-unit k3/k4 corrections by wrapping the pair map and the final scalar map with access to the layer's hstate."""
    def __init__(self, *aa, **kw):
        super().__init__(*aa, **kw)
        f = self.env; t = self.t; card = self
        orig_pair = f['_nonlinear_pair_batch']; orig_scalar = f['nonlinear_scalar']
        def corr(h):
            c3 = AMP * t.tanh(h.to(t.float32) @ head['w3'] + head['b3']).to(t.float64)
            c4 = AMP * t.tanh(h.to(t.float32) @ head['w4'] + head['b4']).to(t.float64)
            cv = VAMP * t.tanh(h.to(t.float32) @ head['wv'] + head['bv']).to(t.float64) if VAMP else None
            c21 = K21AMP * t.tanh(h.to(t.float32) @ head['w21'] + head['b21']).to(t.float64) if K21AMP else None
            return c3, c4, cv, c21
        def pair(mu, C, K21, k4, k22_override=None, metric='identity'):
            if card.call_layer >= FROM and card.cur_h is not None:
                c3, c4, cv, c21 = corr(card.cur_h)
                d = t.diagonal(K21)
                if c21 is not None:
                    s21 = t.sqrt(1 + c21); K21 = K21 * s21[:, None] * s21[None, :]
                K21 = t.diagonal_scatter(K21, d * (1 + c3))
                k4 = k4 * (1 + c4)
                if k22_override is not None:
                    s4 = t.sqrt(1 + c4); k22_override = k22_override * s4[:, None] * s4[None, :]
                    k22_override = t.diagonal_scatter(k22_override, k4)
                if cv is not None:
                    C = t.diagonal_scatter(C, t.diagonal(C) * (1 + cv))
            return orig_pair(mu, C, K21, k4, k22_override, metric)
        def scalar(mu, v, k3, k4):
            if card.cur_h is not None:
                c3, c4, cv, c21 = corr(card.cur_h); k3 = k3 * (1 + c3); k4 = k4 * (1 + c4)
                if cv is not None: v = v * (1 + cv)
            return orig_scalar(mu, v, k3, k4)
        f['_nonlinear_pair_batch'] = pair; f['nonlinear_scalar'] = scalar
        self.cur_h = None
    def step(self, layer, state):
        self.cur_h = state[8]  # hstate from the previous layer (causal)
        return super().step(layer, state)

def run_net(rank, grad):
    card = CumHeadCard(torch, weights(rank), p, a, c, e)
    if grad: pred = card.run(True)
    else:
        with torch.no_grad(): pred = card.run(False)
    return ((pred[15].to(torch.float64) - truth(rank)) ** 2).mean()
def evaluate(ranks, label):
    losses = {r: float(run_net(r, False)) for r in ranks}
    print(json.dumps(dict(eval=label, mean=float(np.mean(list(losses.values()))), geo_mean=math.exp(np.mean([math.log(v) for v in losses.values()])), n=len(losses))), flush=True)
    return losses
def save(name):
    torch.save(dict(p={k: v.detach().clone() for k, v in p.items()}, a=a.detach().clone(), c=c.detach().clone(), e={k: v.detach().clone() for k, v in e.items()},
                    head={k: v.detach().clone() for k, v in head.items()}, amp=AMP, vamp=VAMP, k21amp=K21AMP, from_layer=FROM), OUT / name)
log = dict(args=vars(args), epochs=[]); log['valid_before'] = evaluate(VALID, 'valid_before'); json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
opt = torch.optim.Adam(params, lr=args.lr); steps_total = args.epochs * math.ceil(len(TRAIN) / args.batch); step = 0
for epoch in range(args.epochs):
    order = rng.permutation(TRAIN).tolist(); t0 = time.time(); train_losses = []
    for i in range(0, len(order), args.batch):
        batch = order[i:i + args.batch]
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / steps_total)) * (min(1.0, (step + 1) / args.warmup) if args.warmup else 1.0)
        for g in opt.param_groups: g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        for r in batch:
            loss = run_net(r, True) * 1e8 / len(batch); loss.backward(); train_losses.append(float(loss) * len(batch) / 1e8)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
        print(json.dumps(dict(epoch=epoch, update=step, lr=lr, recent_train_mean=float(np.mean(train_losses[-20:])), head_norms=[round(float(head[k].norm()), 4) for k in ('w3', 'w4', 'wv', 'w21')], elapsed=round(time.time() - t0, 1))), flush=True)
    save(f'params_epoch{epoch}.pt'); v = evaluate(VALID, f'valid_epoch{epoch}')
    log['epochs'].append(dict(epoch=epoch, train_mean=float(np.mean(train_losses)), valid=v, wall=round(time.time() - t0, 1))); json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
print(json.dumps(dict(done=True)), flush=True)
