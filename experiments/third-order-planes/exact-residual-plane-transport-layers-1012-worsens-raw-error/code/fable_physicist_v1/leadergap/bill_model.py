#!/usr/bin/env python3
"""TAG fable-physicist (LG-3): native bill ESTIMATE for the arch_screen variants. Per-layer, per-function cost coefficients from the
analyst's native bank census (fable_analyst_v1/bill_by_layer_candidate_r35.json) divided by the bank's torch counts (screen2 cfg
'bank'); a variant's bill change = sum over layers of coefficient x (variant feature - S2 feature). Features: young push = prev x young
columns; young read = response x live x young; old reconstruct and old inner = response x old; old push = prev (+ joined x prev at
joins); star and covariance pushes fixed. Terminal cuts scale the L15 row-proportional costs by the kept-row fraction.
usage: bill_model.py <screen jsonl ...>"""
import sys, json, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
cen = json.loads((P2 / 'fable_analyst_v1/bill_by_layer_candidate_r35.json').read_text()); T = cen['total']; pf = cen['per_layer_fn']
cost = lambda l, f: pf.get(f'{l}|{f}', 0) / T          # share of the bank bill
rows = [json.loads(x) for p in sys.argv[1:] for x in open(p)]
recs = {(r['rank'], r['cfg']): r for r in rows}
def feats(r):
    out = {}
    for l in range(1, 16):
        c = r['counts'][str(l)]; young = c['total'] - c['old']
        resp, live = c.get('count1', 1024), c.get('count0', 1024)
        out[l] = dict(push=c['prev'] * young, read=resp * live * young, rec=resp * c['old'], inner=resp * c['old'],
                      opush=c['prev'] * (384 + c['joined']) if c['old'] else 0)
    return out
FN = dict(push='_push_bank_shared_newborn', read='_shared_old384_read', rec='_slr_reconstruct', inner='_slr_inner', opush='_shared_old384_push')
for rank in sorted({k[0] for k in recs}):
    bank, s2 = recs.get((rank, 'bank')) or recs.get((35, 'bank')), recs[(rank, 's2')]
    fb, fs = feats(bank), feats(s2)
    coef = {}
    for k, f in FN.items():
        tot_c = sum(cost(l, f) for l in range(1, 15)); tot_x = sum(fb[l][k] for l in range(1, 15))
        for l in range(1, 15):
            coef[(l, k)] = cost(l, f) / fb[l][k] if fb[l][k] and cost(l, f) else tot_c / max(tot_x, 1)
    l15 = cost(15, '_slr_reconstruct') + cost(15, '_push_bank_shared_newborn') + cost(15, '_flms_star_diagonal')
    for (rk, cfg), r in sorted(recs.items()):
        if rk != rank or cfg in ('bank', 's2'): continue
        if cfg.startswith('term'):
            d = -(1 - r['term_row_frac']) * l15
        else:
            fv = feats(r); d = sum(coef[(l, k)] * (fv[l][k] - fs[l][k]) for l in range(1, 15) for k in FN)
        adj = (1 + r['rel']) * (1 + d / 0.9864) - 1        # S2 bill = 0.9864 x bank bill (r35 native utils .18575/.1883)
        print(json.dumps(dict(rank=rank, cfg=cfg, raw=round(100 * r['rel'], 2), bill_est=round(100 * d / 0.9864, 2), adjusted_est=round(100 * adj, 2))))
