#!/usr/bin/env python3
"""TAG fable-refit (RF-24): HEADS-ONLY retrain of the learned closure's output layers (GNN mean-hook head/head_b[/gain], covariance-edge
output/bias) with the zero-cost physics stack ON or OFF (control), same TRAIN/VALID nets and schedule. Physics: per-row K31 factor
r_a = .85 + .5 mu_a k3_a / v_a^2 and per-unit k4 = radial v^2 + .5 (k3^2/v - s v^2) (layers >= 2 and the final map).
usage: refit_physics.py --tag T --physics 0|1 --ntrain 32 --nvalid 16 --epochs 2 --lr 1e-3 --threads 3 [--gain]"""
import sys, json, math, time, argparse, ast
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--physics', type=int, default=1); ap.add_argument('--ntrain', type=int, default=32); ap.add_argument('--nvalid', type=int, default=16)
ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--threads', type=int, default=3); ap.add_argument('--gain', action='store_true'); ap.add_argument('--batch', type=int, default=4)
args = ap.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
OUT = HERE / args.tag; OUT.mkdir(exist_ok=True)
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
inputs = json.loads((P2 / "codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json").read_text())
TRAIN, VALID = inputs['train'][:args.ntrain], inputs['valid'][:args.nvalid]
text = model.SOURCE.read_text(); tree = ast.parse(text)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_nonlinear_pair_batch'][-1]
src = ast.get_source_segment(text, node)
old = "    T31 = t4[:, None] * rho if k31 else None\n"
assert src.count(old) == 1
src = src.replace(old, "    T31 = (t4 * (_ROWF[0](mu, k3, v) if _ROWF[0] is not None else 1.0))[:, None] * rho if k31 else None\n")
p, a, c, e = model.torch_payload(torch, np, False)
params = [p['head'].requires_grad_(True), p['head_b'].requires_grad_(True), e['output'].requires_grad_(True), e['bias'].requires_grad_(True)]
if args.gain: params.append(p['gain'].requires_grad_(True))
def weights(r): return np.load(STORE / Path(man[r]["path"]).parent / "weights.npy")
def truth(r): return torch.tensor(np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{r:04d}.npz")["mc_post"][15].astype(np.float64))
def card_for(rank):
    card = model.Card(torch, weights(rank), p, a, c, e)
    if args.physics:
        card.env['_ROWF'] = [lambda mu, k3, v: 0.85 + 0.5 * mu * k3 / (v * v)]
        t_ = model.Translate().visit(ast.parse(src)); ast.fix_missing_locations(t_); exec(compile(t_, 'pairrow', 'exec'), card.env)
        pair = card.env['_nonlinear_pair_batch']; scal = card.env['nonlinear_scalar']
        def k4n(k3, v, k4):
            q = k3 * k3 / v; s = q.sum() / (v * v).sum(); kn = k4 + 0.5 * (q - s * v * v); return torch.maximum(kn, 0.1 * k4)
        def wpair(mu, C, K21, k4, k22_override=None, metric='identity'):
            if card.call_layer >= 2:
                k4 = k4n(torch.diagonal(K21), torch.diagonal(C), k4)
                if k22_override is not None: k22_override = torch.diagonal_scatter(k22_override, k4)
            return pair(mu, C, K21, k4, k22_override, metric)
        card.env['_nonlinear_pair_batch'] = wpair; card.env['nonlinear_scalar'] = lambda mu, v, k3, k4: scal(mu, v, k3, k4n(k3, v, k4))
    return card
def loss(rank, grad):
    card = card_for(rank)
    if grad: pred = card.run(True)
    else:
        with torch.no_grad(): pred = card.run(False)
    return ((pred[15].to(torch.float64) - truth(rank)) ** 2).mean()
def evaluate(label):
    L = {r: float(loss(r, False)) for r in VALID}
    print(json.dumps(dict(eval=label, mean=float(np.mean(list(L.values()))), n=len(L))), flush=True); return L
log = dict(args=vars(args), epochs=[]); log['valid_before'] = evaluate('valid_before'); json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
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
