#!/usr/bin/env python3
"""TAG fable-physicist (LG-9, = LG-5 + cfg starcols:<L0>-<L1>:<eps> = the audit card's exact star-as-columns transport (AU_STAR_COLS) only at layers L0..L1, strength eps; LG-5 = LG-2 + cfg term32:<zc> = the native build_l15_rowcut.py row set: select(zc - |z|, 0), rounded up to 32): architecture screen on the torch S2 card (model_bank_audit + K31 row .85/.5 L1-14 + b-fix .1 +
every3 sketch push; learned ON, no refit, CPU). Each cfg records final MSE vs truthpost mc_post[15] and per-layer counts
(alive / response / births from select slots 0/1/2, total and old columns from the push) for the bill model.
cfgs: s2 | sched:<name> (birth-into-frame schedules) | sel:<slot>:<thr>:<L0> (select threshold at layers >= L0; slot 0 alive,
1 response rows, 2 births) | term:<zc> (terminal L15: CP + star parts of k3 kept only on rows with |mu/s| < zc)
usage: arch_screen.py <ranks> <cfgs> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE.parent; P2 = PH.parent
sys.path.insert(0, str(PH)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_audit as model
ranks = [int(x) for x in sys.argv[1].split(',')]; cfgs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3])); out = Path(sys.argv[4])
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
SCHED = {'every': {l: [l - 5] for l in range(5, 15)}, 'every3': {7: [0, 1, 2], 10: [3, 4, 5], 13: [6, 7, 8]},
         'dir14': {7: [0, 1, 2], 10: [3, 4, 5], 13: [6, 7, 8], 14: [9, 10, 11, 12, 13]},
         'dir13': {7: [0, 1, 2], 10: [3, 4, 5], 13: [6, 7, 8, 9, 10, 11, 12], 14: [13]},
         'dir10': {7: [0, 1, 2], 10: [3, 4, 5, 6, 7, 8, 9], 11: [10], 12: [11], 13: [12], 14: [13]}}
def make_push(qr, joins, log):
    def push(torch, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c for L_, cs in joins.items() if L_ < layer for c in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            Omega = w32[previous_indices, :rank]
            Y = J @ (weight[:, None] * (J.T @ Omega))
            Qp = Q[previous_indices, :] if old_columns else None
            if old_columns: Y = Y + Qp @ (core @ (Qp.T @ Omega))
            Qn = qr(torch, Y)
            Fj = Qn.T @ J; joined = Fj @ (weight[:, None] * Fj.T)
            if old_columns:
                overlap = Qn.T @ Qp
                F = torch.cat((overlap @ F, Fj), dim=1); core = overlap @ (core @ overlap.T) + joined
            else:
                F, core = Fj, joined
            basis, old_columns = Qn, stop
            log[layer] = dict(prev=len(previous_indices), total=int(p_full.shape[1]), old=old_columns, joined=stop - start)
        elif old_columns:
            basis = Q[previous_indices, :]
            log[layer] = dict(prev=len(previous_indices), total=int(p_full.shape[1]), old=old_columns, joined=0)
        else:
            log[layer] = dict(prev=len(previous_indices), total=int(p_full.shape[1]), old=0, joined=0)
            return wp @ p_full[previous_indices, :], None, F, core, 0
        U = wp @ basis
        return torch.cat((U @ F, wp @ p_full[previous_indices, old_columns:]), dim=1), U, F, core, old_columns
    return push
def k31_row(b0, b1):
    def fn(card, layer, mu, Cpre, vv, radial, k21, k4, k22, parts):
        t = card.t; s = vv.sqrt(); eye = t.eye(vv.shape[0], dtype=t.bool)
        k3 = t.diagonal(k21)
        X = radial * (vv * ((b0 - 1) + b1 * mu * k3 / (vv * vv)))[:, None] * Cpre
        return k21, k4, k22, (X / (s[:, None] ** 3 * s[None, :])).masked_fill(eye, 0.0).to(t.float32)
    return fn
LAYER_SEL = {}
_orig_select = model.Card.select
STAR_LAYERS = set(); STAR_EPS = [1.0]
_orig_step = model.Card.step
def _step(self, layer, state):
    model.AU_STAR_COLS = layer in STAR_LAYERS; model.AU_STAR_EPS = STAR_EPS[0]
    return _orig_step(self, layer, state)
model.Card.step = _step
def _select(self, score):
    thr = LAYER_SEL.get((self.call_layer, self.slot))
    model.AU_SEL = {} if thr is None else {self.slot: thr}
    return _orig_select(self, score)
model.Card.select = _select
p, a, c, e = model.torch_payload(torch, np, False)
def run(w, TP, cfg, term_parts=None):
    LAYER_SEL.clear(); STAR_LAYERS.clear(); STAR_EPS[0] = 1.0; sched = 'every3'; sup15 = None
    kind, _, arg = cfg.partition(':')
    if kind == 'sched': sched = arg
    elif kind == 'sel':
        slot, thr, L0 = arg.split(':')
        for l in range(int(L0), 15): LAYER_SEL[(l, int(slot))] = float(thr)
    elif kind == 'starcols':
        span, eps_ = arg.split(':'); l0, l1 = (int(x) for x in span.split('-'))
        STAR_LAYERS.update(range(l0, l1 + 1)); STAR_EPS[0] = float(eps_)
    elif kind == 'term32':
        zc = float(arg); score = zc - (term_parts['mu'] / term_parts['v'].sqrt()).abs()
        count = min(1024, ((int((score >= 0).sum()) + 31) // 32) * 32)
        keep = torch.zeros_like(score); keep[torch.argsort(-score, stable=True)[:count]] = 1.0
        sup15 = {'k3': keep * (term_parts['cp_old'] + term_parts['cp_young'] + term_parts['star']) + term_parts['bterm']}
        term_frac = float(keep.mean())
    elif kind == 'term':
        zc = float(arg); z = (term_parts['mu'] / term_parts['v'].sqrt()).abs()
        keep = (z < zc).to(torch.float64)
        sup15 = {'k3': keep * (term_parts['cp_old'] + term_parts['cp_young'] + term_parts['star']) + term_parts['bterm']}
        term_frac = float(keep.mean())
    model.PHYS_SUPPLY = {l: {'fn': k31_row(0.85, 0.5)} for l in range(1, 15)}
    if sup15 is not None: model.PHYS_SUPPLY[15] = sup15
    if cfg == 'bank': sched = 'every'; model.PHYS_SUPPLY = None
    model.BFIX_GAMMA = 0.0 if cfg == 'bank' else 0.1; model.AU_STATS = {}; model.OT_CAPTURE = {} if cfg == 's2' else None
    assert model.BFIX_MODE == 'partition' and model.BFIX_MEAN == 'm'
    card = model.Card(torch, w, p, a, c, e); log = {}
    card.shared['push'] = make_push(card.shared['_qr'], SCHED[sched], log)
    t0 = time.time()
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    rec = dict(cfg=cfg, mse=float(((pred[15] - TP[15]) ** 2).mean()), wall=round(time.time() - t0, 1),
               counts={l: dict(model.AU_STATS.get(l, {}), **log.get(l, {})) for l in range(16)})
    if kind in ('term', 'term32'): rec['term_row_frac'] = term_frac
    return rec, (model.OT_CAPTURE or {}).get(15)
with out.open('a') as fo:
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
        TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
        base, parts = run(w, TP, 's2')
        base['rank'] = rank; base['rel'] = 0.0
        fo.write(json.dumps(base) + '\n'); fo.flush(); print(json.dumps({k: base[k] for k in ('rank', 'cfg', 'mse', 'wall')}), flush=True)
        for cfg in cfgs:
            if cfg == 's2': continue
            rec, _ = run(w, TP, cfg, parts)
            rec['rank'] = rank; rec['rel'] = rec['mse'] / base['mse'] - 1
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            print(json.dumps({k: rec[k] for k in ('rank', 'cfg', 'mse', 'rel', 'wall') + (('term_row_frac',) if 'term_row_frac' in rec else ())}), flush=True)
