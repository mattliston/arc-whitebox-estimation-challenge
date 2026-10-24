#!/usr/bin/env python3
"""TAG fable-analyst (AN-33): LAYER-RESOLVED bill dials, torch research card (learned ON, no refit): cheapen a cost center ONLY at early
layers and measure the closed-loop final MSE on each net, with the per-layer counts that drive the bill (alive, response rows, births,
signed columns, young columns). spec terms joined by '+', each '<dial>@<L0>-<L1>':
  birth<f>   birth selection floor 0.01 -> f (select slot 2: fewer newborn columns)
  resp<f>    response-row floor 0.01 -> f (select slot 1: fewer rows in the old/young plane read)
  alive<f>   alive floor 0.01 -> f (select slot 0)
  stardrop   star push (_flms_star, 3 products) replaced by zero
  stardiag   star push reduced to its diagonal (_flms_star_diagonal, 1 product)
  readlr<r>  the atoms' (2,1)-plane read (response rows x alive columns, old tier + young cohorts) replaced by its best rank-r
             approximation (deployable as a randomized sketch: about 3 r n cols flops instead of n^2 cols)
  starrf<r>  deployable version: A -> Q Q^T A with Q = orth(A Omega), Omega a fixed Gaussian n x r sketch (one pass, seed = layer)
  starrfp<r> the same with one power iteration: Q = orth(A A^T A Omega)
  starlr<r>  star matrix A replaced by its best rank-r approximation before the push (deployable as a randomized range finder:
             about 6 n^2 r flops instead of 3 products of n^3)
usage: early_dials.py <ranks> <specs> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE0 = Path(__file__).resolve().parent; P2 = HERE0.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
HERE = HERE0
ranks = [int(x) for x in sys.argv[1].split(',')]; specs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3])); OUT = HERE / sys.argv[4]
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
def parse(spec):
    cfg = dict(floors={}, star={})
    if spec == 'base': return cfg
    for term in spec.split('+'):
        dial, rng = term.split('@'); L0, L1 = [int(x) for x in rng.split('-')]
        for l in range(L0, L1 + 1):
            if dial.startswith('birth'): cfg['floors'][(l, 2)] = float(dial[5:])
            elif dial.startswith('resp'): cfg['floors'][(l, 1)] = float(dial[4:])
            elif dial.startswith('alive'): cfg['floors'][(l, 0)] = float(dial[5:])
            elif dial in ('stardrop', 'stardiag') or dial.startswith('starlr') or dial.startswith('starrf'): cfg['star'][l] = dial
            elif dial.startswith('readlr'): cfg.setdefault('read', {})[l] = int(dial[6:])
            else: raise ValueError(term)
    return cfg
def run(rank, cfg, w, truth):
    card = model.Card(torch, w, p, a, c, e); t = torch
    counts = {}
    def sel(score, card=card):
        l, slot = card.call_layer, card.slot
        floor = cfg['floors'].get((l, slot), 0.01)
        count = min(1024, ((int((score.detach() >= floor).sum()) + 31) // 32) * 32)
        ids = t.argsort(-score.detach(), stable=True)[:count]
        alive = t.zeros_like(score).scatter(0, ids, t.ones(count, dtype=score.dtype))
        card.slot += 1; counts.setdefault(l, {})[('alive', 'resp', 'births')[slot]] = count
        return ids, alive
    card.select = sel
    orig_star = card.env['_flms_star']; orig_diag = card.env['_flms_star_diagonal']
    def star(w_, A):
        mode = cfg['star'].get(card.call_layer)
        if mode == 'stardrop': return torch.zeros((w_.shape[0], w_.shape[0]), dtype=torch.float64)
        if mode == 'stardiag': return torch.diag(orig_diag(w_, A).to(torch.float64))
        if mode is not None and mode.startswith('starrf'):
            power = mode.startswith('starrfp'); r = int(mode[7:] if power else mode[6:]); A64 = A.to(torch.float64)
            Om = torch.randn(A.shape[1], r, generator=torch.Generator().manual_seed(1000 + card.call_layer), dtype=torch.float64)
            Y = A64 @ Om
            if power: Y = A64 @ (A64.T @ Y)
            Q = torch.linalg.qr(Y, mode='reduced')[0]
            return orig_star(w_, (Q @ (Q.T @ A64)).to(A.dtype))
        if mode is not None and mode.startswith('starlr'):
            r = int(mode[6:]); U_, S_, Vh_ = torch.linalg.svd(A.to(torch.float64), full_matrices=False)
            return orig_star(w_, ((U_[:, :r] * S_[:r]) @ Vh_[:r]).to(A.dtype))
        return orig_star(w_, A)
    card.env['_flms_star'] = star
    orig_read = card.shared['read']
    def read(*args, **kw):
        plane = orig_read(*args, **kw); r = cfg.get('read', {}).get(card.call_layer)
        if r is None: return plane
        U_, S_, Vh_ = torch.linalg.svd(plane.to(torch.float64), full_matrices=False)
        return ((U_[:, :r] * S_[:r]) @ Vh_[:r]).to(plane.dtype)
    card.shared['read'] = read
    orig_step = card.step
    def step(layer, state):
        out = orig_step(layer, state)
        counts.setdefault(layer, {})['columns'] = int(out[3].numel())
        return out
    card.step = step
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth) ** 2).mean()), counts
with OUT.open('a') as fo:
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
        truth = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
        base = None
        for spec in (['base'] + [s_ for s_ in specs if s_ != 'base']):
            t0 = time.time(); mse, counts = run(rank, parse(spec), w, truth)
            if spec == 'base': base = mse
            rec = dict(rank=rank, spec=spec, mse=mse, ratio=round(mse / base, 5), wall=round(time.time() - t0, 1),
                       counts={str(l): counts[l] for l in sorted(counts)})
            fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps({k: rec[k] for k in ('rank', 'spec', 'ratio', 'wall')}), flush=True)
