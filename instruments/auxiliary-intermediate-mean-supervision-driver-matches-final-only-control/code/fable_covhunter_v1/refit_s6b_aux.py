#!/usr/bin/env python3
"""TAG fable-covhunter (CH-42) = fable-analyst refit_s6b.py (AN-43, copied 2026-09-23 unchanged except the AUX block) + AUXILIARY
INTERMEDIATE-MEAN SUPERVISION behind --aux_w: loss = final MSE + w * sum_{l in aux_layers} MSE_l * (s_15 / s_l), MSE_l = mean over units of
(m_l - truth_l)^2 against the organizer 1e9 means of truth688 (all 16 layers), s_l = the card's own init MSE of layer l averaged over the
first --aux_calib TRAIN nets (so every aux term is commensurate with the final MSE at init). --aux_w 0 gives the original loss exactly.
--smoke runs the smoke test (w=0 loss identity, finite grads at w, one step moves the heads) on 2 TRAIN nets and the per-layer init MSEs on 4
VALID nets, then exits. Outputs go to fable_covhunter_v1/<tag>.
v2 (2026-09-23 07:50): evaluate() grades the FINAL-layer MSE only (v1 returned the aux-inclusive loss at w > 0; the smoke test never
called it) and logs the VALID MSE of each aux layer beside it; each update also logs batch_final (the final-MSE part, comparable to
fable-physicist refit6b's batch_mean), grad_norm (pre-clip) and lr. --smoke now restores EVERY trained tensor before the VALID4 table (v1
restored only head/head_b, so v1's table carried one Adam step on A, cstar and the edges). v1 = refit_s6b_aux_v1_smoke.py.
Original header follows. refit the learned closure on the FIRED S6b physics (s6bcard.py: S2 physics + v3 covariance knobs + L15 k3
row cut zc 3.5, on covhunter's HookCard), starting from S6b's own learned literals (params_epoch2 of refit_s2_full_lr2e4_v2).
Sibling of refit_s2.py (same loop, loss, schedule, checkpoints and .pt format, so fable_porter_v1/build_refit_candidate_v2.py exports it).
Groups: 'heads' = GNN head/head_b + edge output/bias; 'full' adds the birth coefficients (A, cstar) and all edge parameters;
'full+trunk' adds the GNN trunk (msg_w/b, enc0/2/4 w/b, gain) as a separate Adam group at --trunk_lr. '+knobs' (alone or with heads,
full, full+trunk) trains the covariance-knob strengths per band plus the K31 row c0, c1 (s6bknobs.py: 22 float64 scalars, init = the
shipped values, bit-identical to s6bcard) as a third Adam group at --knob_lr with its own grad-norm clip; values go to knobs_epochN.json
(not the .pt, which keeps the exporter's p/a/c/e contract). v2 (no knobs) is kept as refit_s6b_v2_split.py. --mid_eval U evaluates
VALID[:16] after update U and saves params_midU.pt / knobs_midU.json (v3 without it: refit_s6b_v3_knobs.py). Never trained: the feature
normalization (centers, scales, rms; feature 2 has scale 1e-6) and 'first' (unused by gnn_step).
Loss = final-layer MSE vs truthpost; Adam, batches of 4 nets, cosine lr with 3-update warmup, grad-norm clip 1 over all trained tensors;
VALID evaluated before (= the no-retrain control) and after every epoch. Weights are loaded per call (no cache).
--split <json> takes the training ranks from the file's 'train' list (split_train640.json: 641 nets, no VALID40/panel8 net); without it
TRAIN = inputs.json's 160. Truths come from fable_refit_v1/truth688 (organizer 1e9-sample means for all 688 observer_fit ranks,
bit-identical to public_g22 truthpost on its 200 nets). v1 (TRAIN160 only, public_g22 truths) is kept as refit_s6b_v1_train160.py.
usage: refit_s6b.py --tag T [--groups heads|full|full+trunk|knobs|<any>+knobs] [--lr 1e-4] [--trunk_lr 5e-5] [--knob_lr 1e-3]
                    [--split split_train640.json] [--ntrain N] [--nvalid 16] [--epochs 3]
                    [--threads 3] [--batch 4] [--init <params.pt>|none]"""
import sys, json, math, time, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(0, str(HERE))
import numpy as np, torch
import s6bcard, s6bknobs
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--groups', default='full'); ap.add_argument('--ntrain', type=int, default=None)
ap.add_argument('--split', default=None); ap.add_argument('--mid_eval', type=int, default=0)
ap.add_argument('--nvalid', type=int, default=16); ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--trunk_lr', type=float, default=5e-5); ap.add_argument('--knob_lr', type=float, default=1e-3); ap.add_argument('--threads', type=int, default=3); ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--init', default=str(s6bcard.PARAMS))
ap.add_argument('--aux_layers', default='10,11,12,13,14'); ap.add_argument('--aux_w', type=float, default=0.3); ap.add_argument('--aux_calib', type=int, default=8)
ap.add_argument('--smoke', action='store_true')
args = ap.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
parts = args.groups.split('+'); KNOB = 'knobs' in parts; base_groups = '+'.join(x for x in parts if x != 'knobs')
assert base_groups in ('', 'heads', 'full', 'full+trunk') and len(set(parts)) == len(parts), args.groups
OUT = HERE / args.tag; OUT.mkdir(exist_ok=args.smoke)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
base_train = json.loads(Path(args.split).read_text())['train'] if args.split else inputs['train']
TRAIN, VALID = base_train[:args.ntrain], inputs['valid'][:args.nvalid]
assert not set(TRAIN) & set(inputs['valid']), 'training ranks overlap VALID40'
p, a, c, e = s6bcard.payload(torch, params=None if args.init == 'none' else args.init)
params = [p['head'], p['head_b'], e['output'], e['bias']] if base_groups else []
if base_groups in ('full', 'full+trunk'):
    params += [a, c] + [e[k] for k in e if k not in ('output', 'bias')]
TRUNK = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b', 'gain')
trunk = [p[k] for k in TRUNK] if base_groups == 'full+trunk' else []
K = s6bknobs.init(torch) if KNOB else None; knobs = list(K.values()) if KNOB else []
for x in params + trunk + knobs: x.requires_grad_(True)
def weights(r):   # no cache (refit_s2_v1_cachebug.py grew ~3 GB by caching 64 MB per net)
    return np.load(STORE / Path(man[r]['path']).parent / 'weights.npy')
def truth(r): return torch.tensor(np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64))
def truth_all(r): return torch.tensor(np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'].astype(np.float64))
AUX_L = [int(x) for x in args.aux_layers.split(',')] if args.aux_layers else []
LAST, DIAG = {}, {}                                         # LAST['final'] = last final MSE; DIAG[label] = per-net aux-layer VALID MSEs
AUX = dict(w=args.aux_w, layers=AUX_L, scale=None)          # scale = {l: s_l} and 's15', set by calibrate() when w > 0
def forward(rank, grad):
    card = (s6bknobs.card(torch, weights(rank), p, a, c, e, K, fine_ckpt=grad) if KNOB else s6bcard.card(torch, weights(rank), p, a, c, e, fine_ckpt=grad))
    if grad: pred = card.run(True)
    else:
        with torch.no_grad(): pred = card.run(False)
    return pred
def layer_mses(pred, rank):
    ta = truth_all(rank)
    return {l: ((pred[l].to(torch.float64) - ta[l]) ** 2).mean() for l in range(16)}
def loss(rank, grad):
    pred = forward(rank, grad)
    final = ((pred[15].to(torch.float64) - truth(rank)) ** 2).mean(); LAST['final'] = float(final)
    if AUX['w'] == 0 or not AUX['layers']:
        return final                                                   # identical to refit_s6b.py
    ta = truth_all(rank); sc = AUX['scale']
    aux = sum(((pred[l].to(torch.float64) - ta[l]) ** 2).mean() * (sc['s15'] / sc[l]) for l in AUX['layers'])
    return final + AUX['w'] * aux
def calibrate(nets):
    per = [layer_mses(forward(r, False), r) for r in nets]
    sc = {l: float(np.mean([float(x[l]) for x in per])) for l in range(16)}; sc['s15'] = sc[15]
    return sc
if AUX['w'] > 0 and AUX['layers'] and not args.smoke:
    AUX['scale'] = calibrate(TRAIN[:args.aux_calib])
    print(json.dumps(dict(aux_calibration={str(k): v for k, v in AUX['scale'].items()}, nets=TRAIN[:args.aux_calib])), flush=True)
if args.smoke:
    import copy
    t0 = time.time(); nets = TRAIN[:2]; out = dict(nets=nets)
    # (1) w = 0: the loss equals the original driver's expression bit for bit
    AUX['w'] = 0.0
    def loss_orig(rank):   # the original refit_s6b.py loss body, verbatim
        card = (s6bknobs.card(torch, weights(rank), p, a, c, e, K) if KNOB else s6bcard.card(torch, weights(rank), p, a, c, e))
        with torch.no_grad(): pred = card.run(False)
        return ((pred[15].to(torch.float64) - truth(rank)) ** 2).mean()
    out['w0_identical'] = {r: bool(float(loss(r, False)) == float(loss_orig(r))) for r in nets}
    # (2) at w: calibrate on the 2 nets, finite gradients; (3) one Adam step on the 2-net batch moves the heads
    AUX['w'] = args.aux_w; AUX['scale'] = calibrate(nets)
    out['calibration'] = {str(k): v for k, v in AUX['scale'].items()}
    head0 = p['head'].detach().clone(); headb0 = p['head_b'].detach().clone(); snap = [x.detach().clone() for x in params + trunk + knobs]
    opt1 = torch.optim.Adam([dict(params=params + trunk + knobs, lr=args.lr)])
    opt1.zero_grad(set_to_none=True); comp = {}
    for r in nets:
        L = loss(r, True) * 1e8 / len(nets); L.backward(); comp[r] = float(L) * len(nets) / 1e8
    gn = {str(i): (float(x.grad.norm()) if x.grad is not None else None) for i, x in enumerate(params + trunk + knobs)}
    out['loss_at_w'] = comp; out['grads_finite'] = all(v is not None and math.isfinite(v) for v in gn.values())
    out['grad_norm_range'] = [min(v for v in gn.values() if v is not None), max(v for v in gn.values() if v is not None)]
    torch.nn.utils.clip_grad_norm_(params + trunk + knobs, 1.0); opt1.step()
    out['head_moved'] = float((p['head'] - head0).abs().max()); out['head_b_moved'] = float((p['head_b'] - headb0).abs().max())
    # per-layer init MSEs on 4 VALID nets (params restored to init first)
    with torch.no_grad():
        for x, x0 in zip(params + trunk + knobs, snap): x.copy_(x0)
    out['valid4_layer_mse'] = {}
    for r in inputs['valid'][:4]:
        lm = layer_mses(forward(r, False), r); out['valid4_layer_mse'][r] = {l: float(v) for l, v in lm.items()}
    out['wall_s'] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True); (OUT / 'smoke.json').write_text(json.dumps(out, indent=1)); sys.exit()
def evaluate(label, nets=None):   # VALID is graded on the final-layer MSE only, whatever w; aux-layer MSEs are logged beside it
    L, D = {}, {}
    for r in (VALID if nets is None else nets):
        pred = forward(r, False); ta = truth_all(r)
        L[r] = float(((pred[15].to(torch.float64) - truth(r)) ** 2).mean())
        D[r] = {l: float(((pred[l].to(torch.float64) - ta[l]) ** 2).mean()) for l in AUX['layers']}
    DIAG[label] = D
    print(json.dumps(dict(eval=label, mean=float(np.mean(list(L.values()))), n=len(L),
                          aux_mean={l: float(np.mean([D[r][l] for r in D])) for l in AUX['layers']})), flush=True); return L
log = dict(args=vars(args), card='s6bknobs (S6b physics, trainable knobs)' if KNOB else 's6bcard (S6b physics)', epochs=[], aux_scale=AUX['scale'], aux_diag=DIAG,
           n_params=int(sum(x.numel() for x in params + trunk + knobs)), knobs_init=s6bknobs.values(K) if KNOB else None)
log['valid_before'] = evaluate('valid_before'); json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
groups = ([dict(params=params, lr=args.lr, base=args.lr)] if params else []) + ([dict(params=trunk, lr=args.trunk_lr, base=args.trunk_lr)] if trunk else []) \
    + ([dict(params=knobs, lr=args.knob_lr, base=args.knob_lr)] if knobs else [])
opt = torch.optim.Adam(groups); steps = args.epochs * math.ceil(len(TRAIN) / args.batch); step = 0
for ep in range(args.epochs):
    order = rng.permutation(TRAIN).tolist(); tl = []; fl = []; t0 = time.time()
    for i in range(0, len(order), args.batch):
        fac = 0.5 * (1 + math.cos(math.pi * step / steps)) * min(1.0, (step + 1) / 3)
        for g in opt.param_groups: g['lr'] = g['base'] * fac
        opt.zero_grad(set_to_none=True)
        for r in order[i:i + args.batch]:
            L = loss(r, True) * 1e8 / args.batch; L.backward(); tl.append(float(L) * args.batch / 1e8); fl.append(LAST['final'])
        gn = float(torch.nn.utils.clip_grad_norm_(params + trunk, 1.0)) if params + trunk else None
        if knobs: torch.nn.utils.clip_grad_norm_(knobs, 1.0)
        opt.step(); step += 1
        if args.mid_eval and step == args.mid_eval:
            log['mid'] = dict(update=step, valid16=evaluate(f'valid16_mid_update{step}', inputs['valid'][:16]), knobs=s6bknobs.values(K) if KNOB else None)
            json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
            torch.save(dict(p={k: t.detach().clone() for k, t in p.items()}, a=a.detach().clone(), c=c.detach().clone(), e={k: t.detach().clone() for k, t in e.items()}), OUT / f'params_mid{step}.pt')
            if KNOB: (OUT / f'knobs_mid{step}.json').write_text(json.dumps(s6bknobs.values(K), indent=1))
        nb = len(order[i:i + args.batch])
        print(json.dumps(dict(epoch=ep, update=step, batch_final=float(np.mean(fl[-nb:])), batch_loss=float(np.mean(tl[-nb:])),
                              final_recent=float(np.mean(fl[-8:])), train_recent=float(np.mean(tl[-8:])), grad_norm=gn,
                              lr=round(opt.param_groups[0]['lr'], 9), elapsed=round(time.time() - t0))), flush=True)
    v = evaluate(f'valid_epoch{ep}'); log['epochs'].append(dict(epoch=ep, train=float(np.mean(tl)), train_final=float(np.mean(fl)), valid=v, knobs=s6bknobs.values(K) if KNOB else None))
    json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
    if KNOB: (OUT / f'knobs_epoch{ep}.json').write_text(json.dumps(s6bknobs.values(K), indent=1))
    torch.save(dict(p={k: t.detach().clone() for k, t in p.items()}, a=a.detach().clone(), c=c.detach().clone(), e={k: t.detach().clone() for k, t in e.items()}), OUT / f'params_epoch{ep}.pt')
