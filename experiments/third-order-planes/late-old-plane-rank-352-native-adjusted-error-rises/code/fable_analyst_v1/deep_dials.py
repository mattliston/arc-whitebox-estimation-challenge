#!/usr/bin/env python3
"""TAG fable-analyst (AN-37): DEEP-LAYER bill dials and their zero-cost compensations on the S2 torch card (s2card.py: K31 row, b-fix .1,
every3 joins at L7/L10/L13; learned ON, no refit). Final-layer MSE per net plus per-layer counts (alive, response rows, births, columns,
tier rank). Terms joined by '+':
  rank<r>@<J>        tier frame rank r at the every3 join layer J (7, 10 or 13); the rank persists until the next join
  oldscale<d>@a-b    compensation: old-tier part of the atoms' plane read x (1 + d) at layers a-b
  starlr<r>@a-b      star matrix truncated to rank r (exact SVD) before its push
  starscale<d>@a-b   compensation: star push output x (1 + d)
  resp<f>@a-b        response-row floor 0.01 -> f;  respall@a-b: every alive row is a response row
  alive<f>@a-b       live-column (alive) floor 0.01 -> f
  birth<f>@a-b       birth floor 0.01 -> f
  readscale<d>@a-b   compensation: whole atoms' plane read x (1 + d)
usage: deep_dials.py <ranks> <specs> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import s2card
model = s2card.model
ranks = [int(x) for x in sys.argv[1].split(',')]; specs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3])); OUT = HERE / sys.argv[4]
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
def parse(spec):
    cfg = dict(rank={}, floors={}, star={}, starscale={}, oldscale={}, readscale={})
    if spec == 'base': return cfg
    for term in spec.split('+'):
        dial, rng = term.split('@')
        if dial.startswith('rank'): cfg['rank'][int(rng)] = int(dial[4:]); continue
        L0, L1 = ([int(x) for x in rng.split('-')] + [None])[:2]; L1 = L0 if L1 is None else L1
        for l in range(L0, L1 + 1):
            if dial.startswith('oldscale'): cfg['oldscale'][l] = float(dial[8:])
            elif dial.startswith('starscale'): cfg['starscale'][l] = float(dial[9:])
            elif dial.startswith('starlr'): cfg['star'][l] = int(dial[6:])
            elif dial == 'respall': cfg['floors'][(l, 1)] = -0.5
            elif dial.startswith('resp'): cfg['floors'][(l, 1)] = float(dial[4:])
            elif dial.startswith('alive'): cfg['floors'][(l, 0)] = float(dial[5:])
            elif dial.startswith('birth'): cfg['floors'][(l, 2)] = float(dial[5:])
            elif dial.startswith('readscale'): cfg['readscale'][l] = float(dial[9:])
            else: raise ValueError(term)
    return cfg
def make_push_ranked(qr, joins, rank_at):
    def push(torch_, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [cc for L_, cs in joins.items() if L_ < layer for cc in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            r = min(rank_at.get(layer, rank), int(previous_indices.numel()))   # a frame cannot exceed the active rows (r = active: exact)
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            Omega = w32[previous_indices, :r]
            Y = J @ (weight[:, None] * (J.T @ Omega))
            Qp = Q[previous_indices, :] if old_columns else None
            if old_columns: Y = Y + Qp @ (core @ (Qp.T @ Omega))
            Qn = qr(torch_, Y)
            Fj = Qn.T @ J; joined = Fj @ (weight[:, None] * Fj.T)
            if old_columns:
                overlap = Qn.T @ Qp
                F = torch_.cat((overlap @ F, Fj), dim=1); core = overlap @ (core @ overlap.T) + joined
            else:
                F, core = Fj, joined
            basis, old_columns = Qn, stop
        elif old_columns:
            basis = Q[previous_indices, :]
        else:
            return wp @ p_full[previous_indices, :], None, F, core, 0
        U = wp @ basis
        return torch_.cat((U @ F, wp @ p_full[previous_indices, old_columns:]), dim=1), U, F, core, old_columns
    return push
def run(rank, cfg, w, truth):
    card = s2card.card(torch, w, p, a, c, e); t = torch; counts = {}
    if cfg['rank']: card.shared['push'] = make_push_ranked(card.shared['_qr'], s2card.EVERY3, cfg['rank'])
    def sel(score, card=card):
        l, slot = card.call_layer, card.slot
        floor = cfg['floors'].get((l, slot), 0.01)
        count = min(1024, ((int((score.detach() >= floor).sum()) + 31) // 32) * 32)
        ids = t.argsort(-score.detach(), stable=True)[:count]
        alive = t.zeros_like(score).scatter(0, ids, t.ones(count, dtype=score.dtype))
        card.slot += 1; counts.setdefault(l, {})[('alive', 'resp', 'births')[slot]] = count
        return ids, alive
    card.select = sel
    orig_star = card.env['_flms_star']
    def star(w_, A):
        l = card.call_layer; r = cfg['star'].get(l)
        if r is not None:
            U_, S_, Vh_ = torch.linalg.svd(A.to(torch.float64), full_matrices=False); A = ((U_[:, :r] * S_[:r]) @ Vh_[:r]).to(A.dtype)
        out = orig_star(w_, A)
        return out * (1 + cfg['starscale'][l]) if l in cfg['starscale'] else out
    card.env['_flms_star'] = star
    orig_read = card.shared['read']
    def read(torch_, pp, lam, response_indices, live_indices, U, F, old_columns):
        l = card.call_layer; so = cfg['oldscale'].get(l); sr = cfg['readscale'].get(l)
        if so is None or old_columns == 0:
            out = orig_read(torch_, pp, lam, response_indices, live_indices, U, F, old_columns)
        else:
            left, right = pp[response_indices, :], pp[live_indices, :]
            old_left = left[:, :old_columns]
            old_plane = ((old_left * old_left) @ (F * lam[:old_columns][None, :]).T) @ U[live_indices, :].T
            yl, yr = left[:, old_columns:], right[:, old_columns:]
            out = (yl * yl) @ (yr * lam[old_columns:]).T + (1 + so) * old_plane
        return out * (1 + sr) if sr is not None else out
    card.shared['read'] = read
    orig_step = card.step
    def step(layer, state):
        out = orig_step(layer, state)
        counts.setdefault(layer, {})['columns'] = int(out[3].numel()); counts[layer]['tier_rank'] = int(out[10].shape[0]) if out[10].numel() else 0
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
            rec = dict(rank=rank, spec=spec, mse=mse, ratio=round(mse / base, 5), wall=round(time.time() - t0, 1), counts={str(l): counts[l] for l in sorted(counts)})
            fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps({k: rec[k] for k in ('rank', 'spec', 'ratio', 'wall')}), flush=True)
