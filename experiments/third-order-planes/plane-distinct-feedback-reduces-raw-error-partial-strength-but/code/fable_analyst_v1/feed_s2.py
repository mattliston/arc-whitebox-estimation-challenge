#!/usr/bin/env python3
"""TAG fable-analyst (AN-39): the plane-to-distinct FEED of the third-order kernel, ONE-STEP, on the S2 torch card (s2card.py: K31 row,
b-fix .1, every3 joins; learned ON, no refit). Pattern (0,1,0) only:
  1/2 K21pre[u,w] (-alpha_u pdf_u/sigma_u) Phi(alpha_w) (sigma_s Phi(alpha_s)) rho_us,  u, w, s distinct.
Formulas are IMPORTED from public_g29_bond_kernel_v1 (not re-derived): plane_feed_oracle.columns / feed_reads (k3 read, 3 GEMMs) and
plane_feed_plane_read.plane_read_first (full next-layer plane read P, 9 GEMMs; diag P = the k3 read).
Closed loop: at source layer l the feed is computed from the card's own pre-state at l (mu, Cpre, the full pre-plane k21 = atoms read +
star push + b terms, diagonal zeroed; it already holds the feed injected from l-1) with W_{l+1}, and strength x delta is added at READ layer
l+1 to the star push (the star enters the card only through k21), or, at read layer 15, to the final k3 (_flms_star_diagonal).
Specs: base | k3@a-b[xS] | plane@a-b[xS]   a-b = READ layers (source = read - 1), S = strength (default 1).
env FEED_CHECK=1: at every plane source layer also run the k3 read and record |diag P - k3 read| / |k3 read| (certificate link).
usage: feed_s2.py <ranks> <specs> <threads> <out.jsonl>"""
import sys, os, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P2 / 'public_g29_bond_kernel_v1'))
import numpy as np, torch
import s2card
import plane_feed_oracle as pfo
import plane_feed_plane_read as ppr
model = s2card.model
ranks = [int(x) for x in sys.argv[1].split(',')]; specs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3])); OUT = HERE / sys.argv[4]
CHECK = os.environ.get('FEED_CHECK') == '1'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
PAT = (0, 1, 0, 1.0)
CPU = torch.device('cpu')
def parse(spec):
    if spec == 'base': return None
    form, rest = spec.split('@'); rng, s = (rest.split('x') + ['1'])[:2]; L0, L1 = [int(x) for x in rng.split('-')]
    assert form in ('k3', 'plane') and 2 <= L0 <= L1 <= 15, spec          # read 15 is always the k3 read (final scalar layer)
    return dict(form=form, reads=set(range(L0, L1 + 1)), s=float(s))
def feed(form, W, mu, Cpre, K21):
    """(0,1,0) feed of the source pre-state into the next layer: k3 read (1024,) or plane read P (1024, 1024), float64."""
    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64)
    sd, d, b, a_ = pfo.columns(mu.numpy(), np.maximum(torch.diagonal(Cpre).numpy(), 1e-30), pfo.TMAX)
    sdT = T(sd); rho = Cpre / torch.outer(sdT, sdT); rho.fill_diagonal_(0.)
    K = K21.clone(); K.fill_diagonal_(0.)
    if form == 'k3':
        r = pfo.feed_reads(W, K, {0: None, 1: rho}, [T(x) for x in d], [T(x) for x in b], [T(x) for x in a_], [PAT], CPU)[PAT[:3]]
        return T(r), None
    P = ppr.plane_read_first(W, K, T(d[1]), T(b[0]), T(a_[1]), rho, 1, 1.0)
    chk = None
    if CHECK:
        r = pfo.feed_reads(W, K, {0: None, 1: rho}, [T(x) for x in d], [T(x) for x in b], [T(x) for x in a_], [PAT], CPU)[PAT[:3]]
        chk = float(np.linalg.norm(torch.diagonal(P).numpy() - r) / np.linalg.norm(r))
    return P, chk
def run(rank, cfg, w, truth):
    card = s2card.card(torch, w, p, a, c, e); stats = {}; k3rms = {}
    if cfg is not None:
        pending = {}
        orig_pair, orig_star, orig_sdiag = card.env['_nonlinear_pair_batch'], card.env['_flms_star'], card.env['_flms_star_diagonal']
        def pair(mu, C, K21, *args, **kw):
            l = card.call_layer; L = l + 1
            k3rms[l] = float(torch.diagonal(K21.detach()).pow(2).mean().sqrt())
            if L in cfg['reads'] and L not in pending:
                form = 'k3' if (cfg['form'] == 'k3' or L == 15) else 'plane'
                delta, chk = feed(form, card.ws[L], mu.detach(), C.detach(), K21.detach())
                pending[L] = (form, cfg['s'] * delta)
                k3card = torch.diagonal(K21.detach())
                dk3 = delta if form == 'k3' else torch.diagonal(delta)
                rec = dict(src=l, form=form, rms_k3_src=float(k3card.pow(2).mean().sqrt()), rms_dk3_read=float(cfg['s'] * dk3.pow(2).mean().sqrt()))
                if form == 'plane':
                    off = delta - torch.diag(torch.diagonal(delta)); Koff = K21.detach() - torch.diag(k3card)
                    rec.update(rms_offdiag_read=float(cfg['s'] * off.pow(2).mean().sqrt()), rms_offdiag_src=float(Koff.pow(2).mean().sqrt()))
                if chk is not None: rec['diag_vs_k3read_rel'] = chk
                stats[L] = rec
            return orig_pair(mu, C, K21, *args, **kw)
        def star(w_, A):
            out = orig_star(w_, A); L = card.call_layer
            if L in pending:
                form, dl = pending.pop(L)
                out = out.to(torch.float64) + (torch.diag(dl) if form == 'k3' else dl)
            return out
        def sdiag(w_, A):
            out = orig_sdiag(w_, A); L = card.call_layer
            if L in pending:
                form, dl = pending.pop(L); assert form == 'k3'
                out = out.to(torch.float64) + dl
            return out
        card.env['_nonlinear_pair_batch'], card.env['_flms_star'], card.env['_flms_star_diagonal'] = pair, star, sdiag
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    if cfg is not None: assert not pending, ('undelivered feed', sorted(pending))
    for L in stats:
        if L in k3rms: stats[L]['rms_k3_read_layer'] = k3rms[L]
    return float(((pred[15] - truth) ** 2).mean()), stats
with OUT.open('a') as fo:
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
        truth = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
        base = None
        for spec in (['base'] + [s_ for s_ in specs if s_ != 'base']):
            t0 = time.time(); mse, stats = run(rank, parse(spec), w, truth)
            if spec == 'base': base = mse
            rec = dict(tag='AN-39 feed_s2 (0,1,0) one-step closed-loop, S2 torch card, learned ON, no refit', rank=rank, spec=spec, mse=mse,
                       ratio=round(mse / base, 5), wall=round(time.time() - t0, 1), check=CHECK, layers={str(k): stats[k] for k in sorted(stats)})
            fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps({k: rec[k] for k in ('rank', 'spec', 'mse', 'ratio', 'wall')}), flush=True)
