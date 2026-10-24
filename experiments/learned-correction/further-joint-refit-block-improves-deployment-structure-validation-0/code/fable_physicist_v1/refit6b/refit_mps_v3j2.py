#!/usr/bin/env python3
"""TAG fable-physicist (09-25: the TAGLINE also records env JOIN_SCHED when set, for s6bcard_loop4js.) (v3j2 = v3j + the MPS lock HELD for the whole run: keep the context-manager reference; v3j and the
engineer's v3 released the ticket at GC right after HOLD). (loopR3 JOINT refit driver, 09-25): versioned copy of fable_engineer_v1/refit_mps_v3.py (E-68) that ALSO trains the
12-dial family (covhunter s6bcard_joint2t) with the learned parts. Changes: default card s6bcard_loop4j (S23 structure + trainable dials),
no LOOP_DIALMOD; --dials (init values, default the round-1 dials fable_covhunter_v1/dials_r1_ep2.json); --dial-lr JSON {dial: lr}
(dials not listed stay frozen); each trained dial is its own Adam group with the same warmup + cosine schedule; dial gradients are
clipped separately (--dial-clip); every update logs the dial values; dials_epoch<N>.json is saved next to params_epoch<N>.pt.
Dials are float64 CPU leaves; the MPS card gets float32 CPU views rebuilt before each card build (autograd casts a gradient to its
leaf's dtype before moving a 0-dim gradient off MPS; certified loop4j_cert_{cpu,mps}_r37.json). Original docstring follows.
TAG fable-engineer (E-68, refit harness v3 for the loop; v3 = v2 + --card: the loop card module file in fable_physicist_v1/refit6b,
default s6bcard_loop3 = the loopR2 card: loop2 (E-67 drop-in) with the s1-zero star): the physicist's refit loop (fable_physicist_v1/refit6b/refit_s6b.py, same
groups / schedule / clip / 1e8-per-batch loss / batch order / valid_before + per-epoch VALID + params_epoch<N>.pt + SPIKE line) on the
physicist's loop card (s6bcard_loop by path; LOOP_DIALMOD picks the dials) with the E-67 drop-in (fine_ckpt=False + pair_series loop
form), on a chosen device:
  --device mps : mps_card (float32 on MPS; float64 CPU islands for the mean-path products; pair-form k3) -- E-68 certified;
  --device cpu : the float64 CPU card.
Extra: --updates N stops after N updates (schedule still spans the epochs). MPS rule: one MPS job on the box at a time -- with --device
mps this script queues itself on mps_lock (FIFO, released at exit); launch behind fable_porter_v1/machine_gate.sh <peak GB> <label>
like every heavy job.
Outputs: <outdir>/<tag>/ (default outdir = the launch directory).
usage: refit_mps_v3.py --device mps|cpu --tag T [--card s6bcard_loop3] [--outdir .] [--groups full|full+P] [--lr 1e-4] [--lr-trunk 5e-5] [--warmup 3] [--warmup-trunk 5]
       [--clip 1.0] [--epochs 1] [--ntrain 160] [--nvalid 16] [--batch 4] [--threads 5] [--resume params.pt] [--updates N] [--eval-only]"""
import os, sys, json, math, time, argparse
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]
for d in ('fable_analyst_v1', 'fable_physicist_v1/refit6b', 'fable_covhunter_v1', 'fable_engineer_v1'): sys.path.insert(0, str(P2 / d))
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('--device', required=True, choices=('cpu', 'mps')); ap.add_argument('--tag', required=True); ap.add_argument('--groups', default='full')
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--lr-trunk', type=float, default=5e-5); ap.add_argument('--warmup', type=int, default=3)
ap.add_argument('--warmup-trunk', type=int, default=5); ap.add_argument('--clip', type=float, default=1.0); ap.add_argument('--epochs', type=int, default=2)
ap.add_argument('--ntrain', type=int, default=160); ap.add_argument('--nvalid', type=int, default=16); ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--threads', type=int, default=5); ap.add_argument('--resume', default=None); ap.add_argument('--updates', type=int, default=None)
ap.add_argument('--eval-only', action='store_true'); ap.add_argument('--card', default='s6bcard_loop4j'); ap.add_argument('--outdir', default='.')
ap.add_argument('--dials', default=str(P2 / 'fable_covhunter_v1/dials_r1_ep2.json')); ap.add_argument('--dial-lr', default='{}'); ap.add_argument('--dial-clip', type=float, default=1.0)
args = ap.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
import importlib.util
_spec = importlib.util.spec_from_file_location('loopcard_physicist_' + args.card, P2 / 'fable_physicist_v1/refit6b' / (args.card + '.py'))
LC = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(LC)
import pair_series_torch as ps
MPS = args.device == 'mps'
JT = LC.JT; DIAL_LR = json.loads(args.dial_lr); assert set(DIAL_LR) <= set(JT.KEYS), DIAL_LR
_dv = json.load(open(args.dials)); DIAL_INIT = {k: float(_dv[k]) for k in JT.KEYS}
DL = JT.make_dials(torch, DIAL_INIT, trainable=() if args.eval_only else tuple(DIAL_LR))   # float64 CPU leaves
def set_dials():   # before EVERY card build; outside the MPS modes
    JT.DIALS_T = {k: DL[k].to(torch.float32) for k in DL} if MPS else DL
if MPS:
    import mps_card as mc, mps_lock                     # the box's one MPS job: FIFO queue (released at exit)
    _MPS_HOLD = mps_lock.hold(f'refit_mps_v3j {args.tag} (pid {os.getpid()})'); _MPS_HOLD.__enter__()   # KEEP the reference: a discarded
    # context manager is garbage-collected at once and its finally deletes the ticket (the v3/v3j lock bug; released at exit via atexit)
F64 = torch.float64
OUT = Path(args.outdir).resolve() / args.tag; OUT.mkdir(exist_ok=True)   # outputs go where the OWNER runs it (default: the launch directory)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
TRAIN, VALID = inputs['train'][:args.ntrain], inputs['valid'][:args.nvalid]
p, a, c, e = LC.payload(torch)
if args.resume:
    sv = torch.load(args.resume)
    with torch.no_grad():
        for k in p: p[k].copy_(sv['p'][k])
        a.copy_(sv['a']); c.copy_(sv['c'])
        for k in e: e[k].copy_(sv['e'][k])
P_, A_, C_, E_ = (mc.to_mps(p), mc.to_mps(a), mc.to_mps(c), mc.to_mps(e)) if MPS else (p, a, c, e)
full = [P_['head'], P_['head_b'], E_['output'], E_['bias'], A_, C_] + [E_[k] for k in E_ if k not in ('output', 'bias')]
TRUNK_KEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')   # never centers/scales/rms/gain/first
trunk = [P_[k] for k in TRUNK_KEYS] if args.groups == 'full+P' else []
assert args.groups in ('full', 'full+P')
for x in full + trunk: x.requires_grad_(not args.eval_only)
def weights(r): return np.load(STORE / Path(man[r]['path']).parent / 'weights.npy')
def truth(r): return torch.tensor(np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64))
def pred15(rank, grad):
    w = weights(rank)
    set_dials()
    if MPS:
        cd = mc.card(LC, torch, w, P_, A_, C_, E_)
        with mc.on_mps(): pred = cd.run(True) if grad else cd.run(False)
        return pred[15].to('cpu').to(F64)
    cd = ps.install(LC.card(torch, w, P_, A_, C_, E_, fine_ckpt=False), torch)
    return (cd.run(True) if grad else cd.run(False))[15].to(F64)
def loss(rank, grad):
    if grad: return ((pred15(rank, True) - truth(rank)) ** 2).mean()
    with torch.no_grad(): return ((pred15(rank, False) - truth(rank)) ** 2).mean()
def backward(L):
    if MPS:
        with mc.on_mps(): torch.autograd.backward(L)   # NOT L.backward(): see mps_card
    else: L.backward()
def evaluate(label):
    L = {r: float(loss(r, False)) for r in VALID}
    print(json.dumps(dict(eval=label, device=args.device, mean=float(np.mean(list(L.values()))), n=len(L))), flush=True); return L
def save(name):
    to_cpu = lambda x, ref: x.detach().to('cpu').to(ref.dtype).clone()
    dst = OUT / name; assert not dst.exists(), dst
    torch.save(dict(p={k: to_cpu(P_[k], p[k]) for k in p}, a=to_cpu(A_, a), c=to_cpu(C_, c), e={k: to_cpu(E_[k], e[k]) for k in e}), dst)
    dd = OUT / name.replace('params_', 'dials_').replace('.pt', '.json'); assert not dd.exists(), dd
    json.dump(JT.values(DL), open(dd, 'w'), indent=1)
TAGLINE = (args.card + ' (S23 structure: nystrom join + star s1=0 + gn_linear_hook; joint2t trainable dials from ' + Path(args.dials).name
           + ', trained ' + (json.dumps(DIAL_LR) if DIAL_LR and not args.eval_only else 'none') + ') + fine_ckpt=False + pair_series loop form'
           + (' + mps_card float32 (E-68)' if MPS else ' (float64 CPU)')
           + (' + per-join old-tier ranks JOIN_SCHED=' + os.environ['JOIN_SCHED'] if os.environ.get('JOIN_SCHED') else ''))
if args.eval_only:
    L = evaluate(f'eval_only_{Path(args.resume).stem if args.resume else "card_literals"}_n{len(VALID)}')
    json.dump(dict(args=vars(args), card=TAGLINE, valid=L), open(OUT / f'eval_n{len(VALID)}_{args.device}.json', 'w'), indent=1); sys.exit(0)
log = dict(args=vars(args), card=TAGLINE, epochs=[], n_params=dict(full=int(sum(x.numel() for x in full)), trunk=int(sum(x.numel() for x in trunk))))
log['valid_before'] = evaluate('valid_before'); base = float(np.mean(list(log['valid_before'].values())))
print(json.dumps(dict(tag=args.tag, card=TAGLINE, device=args.device, groups=args.groups, n_params=log['n_params'], lr=args.lr, lr_trunk=args.lr_trunk if trunk else None, clip=args.clip)), flush=True)
json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
groups = [dict(params=full, lr=args.lr, base=args.lr, warm=args.warmup)]
if trunk: groups.append(dict(params=trunk, lr=args.lr_trunk, base=args.lr_trunk, warm=args.warmup_trunk))
DIALS_TRAINED = [DL[k] for k in DIAL_LR]
for k in DIAL_LR: groups.append(dict(params=[DL[k]], lr=DIAL_LR[k], base=DIAL_LR[k], warm=args.warmup))
opt = torch.optim.Adam([dict(params=g['params'], lr=g['lr']) for g in groups]); steps = args.epochs * math.ceil(len(TRAIN) / args.batch); step = 0
for ep in range(args.epochs):
    order = rng.permutation(TRAIN).tolist(); tl = []; t0 = time.time()
    for i in range(0, len(order), args.batch):
        for pg, g in zip(opt.param_groups, groups):
            pg['lr'] = g['base'] * 0.5 * (1 + math.cos(math.pi * step / steps)) * min(1.0, (step + 1) / g['warm'])
        opt.zero_grad(set_to_none=True); bl = []
        for r in order[i:i + args.batch]:
            L = loss(r, True) * 1e8 / args.batch; backward(L); bl.append(float(L) * args.batch / 1e8)
        tl += bl
        gn = float(torch.nn.utils.clip_grad_norm_(full + trunk, args.clip))
        dgn = float(torch.nn.utils.clip_grad_norm_(DIALS_TRAINED, args.dial_clip)) if DIALS_TRAINED else 0.0
        opt.step(); step += 1
        rec = dict(epoch=ep, update=step, batch_mean=float(np.mean(bl)), train_recent=float(np.mean(tl[-8:])), grad_norm=round(gn, 4),
                   lr=[round(pg['lr'], 9) for pg in opt.param_groups], elapsed=round(time.time() - t0), dial_grad_norm=round(dgn, 4),
                   dials={k: round(v, 6) for k, v in JT.values(DL).items()})
        print(json.dumps(rec), flush=True)
        if np.mean(bl) > 1.3 * base: print(json.dumps(dict(SPIKE=True, update=step, batch_mean=float(np.mean(bl)), valid_before=base)), flush=True)
        if args.updates and step >= args.updates: break
    v = evaluate(f'valid_epoch{ep}'); log['epochs'].append(dict(epoch=ep, train=float(np.mean(tl)), valid=v)); json.dump(log, open(OUT / 'log.json', 'w'), indent=1)
    save(f'params_epoch{ep}.pt')
    if args.updates and step >= args.updates: break
print(json.dumps(dict(done=True)), flush=True)
