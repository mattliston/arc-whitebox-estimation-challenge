#!/usr/bin/env python3
"""TAG fable-physicist (CR-1): SIZE THE PRIZE of the dropped distinct-index carry term (lead, 09-24).
Dense third-cumulant recursion (T = 1024^3 float32, CPU, chunked in place) driven OPEN-LOOP by one unmodified card run
(s6bcard + B2 ep1 literals, learned on, r35): the card's gates g = Phi(alpha), d = phi/sigma, pre covariance C, alive masks,
newborn CP columns (post space, with the learned leg directions), post plane target (pair map) and b channel.
  capture          run the card once, save per-layer inputs to capture_r35.npz
  A0 <lmax>        the card's own bookkeeping, densely: T = the CP bank only (linear gate x alive, + newborn columns);
                   pre plane = T[i,i,j] + the card's star read sp + the card's b terms  -> must equal the card's k21 up to the
                   old-tier truncation (joins at L7/10/13): the VALIDATION arm
  A <lmax>         the full tensor with LINEAR gate transport (bank + pushed star content, all alive-masked), newborn columns,
                   post plane reset to (target - b-channel post plane); pre plane = T[i,i,j] + the card's b terms
  (A and B are FLAWED: the reset overwrites the card's omitted-row plane, where the card keeps the carried bank plane on top of
   a target computed without it; kept for the record only.)
  C0 <lmax>        THE CARD'S BOOKKEEPING, faithfully and densely (the lead's A): T = the bank (linear gate x alive + newborn);
                   the star is RECOMPUTED each layer from this arm's own bank exactly as the card does (rest = target - trace_post
                   - live*(inherited on response rows x alive + diagonal, + born); star = 3 rest^T - 2 diag rest), pushed with the
                   card's _flms_star formula; the star's distinct content is dropped (as the card); pre plane = T[i,i,j] + sp + b terms.
                   Validates against the card's own star and sp before the first old-tier join (L7).
  D0 <lmax>        C0 + the dressing on ALL bank entries (FLAWED: plane entries (i,i,j) have a different transport factor; the
                   dressed plane mass leaks into the next layer's distinct content past the star; kept for the record)
  D1 <lmax>        C0 + the first-order dressing on DISTINCT entries only; plane and diagonal entries keep the card's linear
                   transport (then the card's star): the lead's B
  B <lmax>         A + the first-order dressing on distinct entries: G = g_i g_j g_k + D_ij g_k + D_ik g_j + g_i D_jk,
                   D = (a d) C (a d) (certified: cert_transport.py, d1 ~ E[f'f'f'] within 1%)
Metric per layer (pre plane, L2..14): relative error vs the 8M truth K21 on the kink block (response rows x alive columns,
off-diagonal) and on the alive k3 diagonal; the card's own errors alongside; floors from PLANE_NOISE_FLOORS.json (log-interpolated).
Device: env DEV=mps runs T on the GPU (float32; MPS has no float64, so plane reads are returned to the CPU for the metrics).
usage: dense_carry.py capture <threads> | [DEV=mps] dense_carry.py <A0|A|B|C0|D0|D1> <lmax> <threads>"""
import sys, json, time, math, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; RF = HERE.parent; P2 = HERE.parents[2]
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_covhunter_v1'))
import numpy as np, torch
MODE = sys.argv[1]
torch.set_grad_enabled(False)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
RANK = 35; N = 1024
WTS = np.load(STORE / Path(man[RANK]['path']).parent / 'weights.npy', mmap_mode='r')
CAP = HERE / f'capture_r{RANK}.npz'

if MODE == 'capture':
    torch.set_num_threads(int(sys.argv[2]))
    import covstep as cs, s6bcard
    p0, a0, c0, e0 = s6bcard.payload(torch)
    sv = torch.load(RF / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
    for k in p0: p0[k].copy_(sv['p'][k])
    a0.copy_(sv['a']); c0.copy_(sv['c'])
    for k in e0: e0[k].copy_(sv['e'][k])
    card = s6bcard.card(torch, np.asarray(WTS), p0, a0, c0, e0)
    rec = {}
    def grab(layer, out, ctx):
        if layer < 15:
            aux = ctx['aux']; C, b, stops, p, lam = out[1], out[5], out[12], out[2], out[3]
            k0 = int(stops[-2]) if len(stops) > 1 else 0; npair = int(stops[-1]) - k0
            trace_post = torch.diagonal(C)[:, None] * b[None, :] + 2 * C * b[:, None]
            f64 = lambda x: x.detach().to(torch.float64).numpy()
            rec[layer] = dict(g=f64(aux['gate']), d=f64(aux['pdf'] / aux['sigma']), C=f64(ctx['Cpre']), alive=f64(ctx['alive']),
                              resp=f64(ctx['response_mask']), P=p[:, k0:k0 + npair].detach().to(torch.float32).numpy(),
                              lam=lam[k0:k0 + npair].detach().to(torch.float32).numpy(), tgt_mb=f64(ctx['target'] - trace_post),
                              k21=f64(ctx['k21']), cp=f64(ctx['cp']), sp=f64(ctx['sp']), bterm=f64(ctx['k21'] - ctx['cp'] - ctx['sp']),
                              star=out[4].detach().to(torch.float32).numpy())
        return out
    card.hooks['state_out'] = grab
    t0 = time.time(); pred = card.run(False)
    truth = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{RANK:04d}.npz')['mc_post'][15].astype(np.float64)
    mse = float(((pred.numpy().astype(np.float64)[15] - truth) ** 2).mean())
    flat = {f'{k}_{l}': v for l, r in rec.items() for k, v in r.items()}
    np.savez(CAP, **flat)
    print(json.dumps(dict(capture=str(CAP), layers=len(rec), card_mse=mse, npairs=[int(rec[l]['P'].shape[1]) for l in sorted(rec)],
                          wall=round(time.time() - t0, 1))), flush=True)
    sys.exit(0)

LMAX, TH = int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(TH)
DEV = torch.device(os.environ.get('DEV', 'cpu'))
sync = (lambda: torch.mps.synchronize()) if DEV.type == 'mps' else (lambda: None)
cap = np.load(CAP)
TRUTH = np.load(P2 / 'public_g19_attribution_v1' / f'mps_planes_rank{RANK:04d}.npz')
FL = json.loads((P2 / 'public_g19_attribution_v1' / 'PLANE_NOISE_FLOORS.json').read_text())
fl_pts = sorted((int(k.split('_L')[1]), v['floor']) for k, v in FL.items() if k.startswith('preK21_L'))
def floor(l):
    xs, ys = [x for x, _ in fl_pts], [math.log(y) for _, y in fl_pts]
    return math.exp(float(np.interp(l, xs, ys)))
SLAB = 32
f32 = lambda x: torch.from_numpy(np.array(x, dtype=np.float32)).to(DEV)
T = torch.zeros((N, N, N), dtype=torch.float32, device=DEV)
flat = T.view(-1)
ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
IDX_IIJ = (ii * N * N + ii * N + jj).reshape(-1).to(DEV)
off = (ii != jj).reshape(-1)
IDX_IJI = (ii * N * N + jj * N + ii).reshape(-1)[off].to(DEV)
IDX_JII = (jj * N * N + ii * N + ii).reshape(-1)[off].to(DEV)
off = off.to(DEV)

def push(W):
    Wt = W.T.contiguous(); X = T.view(N, N * N)
    for c0 in range(0, N * N, 131072):
        X[:, c0:c0 + 131072] = Wt @ X[:, c0:c0 + 131072]
    for a0 in range(0, N, SLAB):
        T[a0:a0 + SLAB] = torch.matmul(Wt, T[a0:a0 + SLAB])
    for a0 in range(0, N, SLAB):
        T[a0:a0 + SLAB] = torch.matmul(T[a0:a0 + SLAB], W)

def plane():
    return flat[IDX_IIJ].view(N, N).cpu().to(torch.float64)

def metrics(l, P_pre):
    out = {}
    if l < 2: return out
    tr = TRUTH[f'K21_{l}']; resp = cap[f'resp_{l}'] > .5; alive = cap[f'alive_{l}'] > .5
    blk = np.ix_(resp, alive); offd = ~np.eye(N, dtype=bool)[blk]
    tb = tr[blk] * offd; nt = np.linalg.norm(tb)
    kt = np.diagonal(tr)[alive]; nk = np.linalg.norm(kt)
    for name, M in (('arm', P_pre), ('card', cap[f'k21_{l}'])):
        e = np.linalg.norm(M[blk] * offd - tb) / nt; ek = np.linalg.norm(np.diagonal(M)[alive] - kt) / nk
        out[name] = dict(plane=round(e, 5), plane_nc=round(math.sqrt(max(e * e - floor(l) ** 2, 0)), 5), k3=round(ek, 5))
    out['arm_vs_card_plane'] = round(np.linalg.norm((P_pre - cap[f'k21_{l}'])[blk] * offd) / nt, 5)
    out['floor'] = round(floor(l), 5)
    return out

def flms_star(W, A):   # the card's _flms_star (zero.py), torch float32 on DEV
    wt = W.T; pushed = wt @ A
    res = (2 * ((pushed * wt) @ W) + (wt * wt) @ pushed.T) / 3
    res.diagonal().copy_((pushed * wt * wt).sum(1)); return res
FAITHFUL = MODE in ('C0', 'D0', 'D1'); STAR = None
rel = lambda x, y: float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))
LOG = dict(arm=MODE, lmax=LMAX, layers={}); t00 = time.time()
for l in range(0, LMAX + 1):
    t0 = time.time()
    if l > 0:
        Wl = f32(WTS[l]); push(Wl)
    if FAITHFUL:
        spn = np.zeros((N, N)) if STAR is None else flms_star(Wl, STAR).cpu().double().numpy()
        P_pre = plane().numpy() + cap[f'bterm_{l}'] + spn
    else:
        P_pre = plane().numpy() + cap[f'bterm_{l}'] + (cap[f'sp_{l}'] if MODE == 'A0' else 0)
    rec = metrics(l, P_pre)
    if FAITHFUL and l > 0:
        rec['sp_vs_card'] = rel(spn, cap[f'sp_{l}'])
    if MODE == 'A0':
        cpm = cap[f'cp_{l}'] != 0            # the card fills cp on response rows x live and the diagonal
        rec['A0_vs_card_cp'] = float(np.linalg.norm((plane().numpy() - cap[f'cp_{l}'])[cpm]) / max(np.linalg.norm(cap[f'cp_{l}'][cpm]), 1e-30))
    if l == LMAX:
        LOG['layers'][l] = rec | dict(t=round(time.time() - t0, 1)); print(json.dumps({l: LOG['layers'][l]}), flush=True)
        break
    a = f32(cap[f'alive_{l}']); ag = a * f32(cap[f'g_{l}'])
    if MODE in ('B', 'D0', 'D1'):
        ad = a * f32(cap[f'd_{l}']); D = ad[:, None] * f32(cap[f'C_{l}']) * ad[None, :]
        M = torch.outer(ag, ag) + D
    else:
        M = torch.outer(ag, ag)
    if MODE == 'D1':
        plane_pre = flat[IDX_IIJ].view(N, N).clone()
    for s0 in range(0, N, SLAB):
        s = slice(s0, s0 + SLAB)
        G = ag[s, None, None] * M[None]
        if MODE in ('B', 'D0', 'D1'):
            G += D[s, :, None] * ag[None, None, :] + D[s, None, :] * ag[None, :, None]
        T[s] *= G
    if MODE == 'D1':   # plane and diagonal entries: the card's linear transport (the dressing is distinct-index only)
        lp = (ag[:, None] ** 2 * ag[None, :] * plane_pre).reshape(-1)
        flat[IDX_IIJ] = lp; flat[IDX_IJI] = lp[off]; flat[IDX_JII] = lp[off]
    if FAITHFUL:
        plane_tr = flat[IDX_IIJ].view(N, N).clone()
    Pn, lam = f32(cap[f'P_{l}']), f32(cap[f'lam_{l}'])
    if Pn.shape[1]:
        PT = Pn.T.contiguous()
        for s0 in range(0, N, SLAB):
            s = slice(s0, s0 + SLAB)
            T[s] += torch.matmul((Pn[s] * lam[None, :])[:, None, :] * Pn[None, :, :], PT)
    if FAITHFUL:   # the card's star, from this arm's own bank
        born = flat[IDX_IIJ].view(N, N) - plane_tr
        live = a[:, None] * a[None, :]
        cpm = (f32(cap[f'resp_{l}'])[:, None] * a[None, :]).clone(); cpm.diagonal().fill_(1.0)
        rest = f32(cap[f'tgt_mb_{l}']) - live * (plane_tr * cpm + born)
        STAR = 3 * rest.T - 2 * torch.diag(torch.diagonal(rest))
        rec['star_vs_card'] = rel(STAR.cpu().double().numpy(), cap[f'star_{l}'])
    if MODE in ('A', 'B'):
        R = f32(cap[f'tgt_mb_{l}']) - flat[IDX_IIJ].view(N, N)
        flat.index_add_(0, IDX_IIJ, R.reshape(-1))
        Ro = R.reshape(-1)[off]
        flat.index_add_(0, IDX_IJI, Ro); flat.index_add_(0, IDX_JII, Ro)
    sync(); LOG['layers'][l] = rec | dict(t=round(time.time() - t0, 1))
    print(json.dumps({l: LOG['layers'][l]}), flush=True)
LOG['dev'] = str(DEV); LOG['wall'] = round(time.time() - t00, 1)
(HERE / f'dense_{MODE}_L{LMAX}{"_mps" if DEV.type == "mps" else ""}.json').write_text(json.dumps(LOG))
print(json.dumps(dict(done=MODE, wall=LOG['wall'])), flush=True)
