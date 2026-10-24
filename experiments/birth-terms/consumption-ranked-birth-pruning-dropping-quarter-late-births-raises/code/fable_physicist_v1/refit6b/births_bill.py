#!/usr/bin/env python3
"""TAG fable-physicist (RF6-7): WHOLE-CARD native bill ratio b of budgeted births (births_prune_eval.py) against S2/S6b, calibrated on the
analyst's native per-layer, per-function census of S2 (bill_by_layer_candS2_e2v7_k31r_bfix01_every3_r35.json) and the torch S2 column
counts on r35 (leadergap/screen2.jsonl). Cohorts born at layer >= L0 keep a share `keep` of their columns. The newborn cohort's first
push is paid in full (the native ranking needs that push), and its first read and everything after use only the kept columns.
Per layer: young push ~ prev x (young columns pushed), atoms read ~ resp x live x (young columns read), old reconstruct ~ 1024 x 384 x old,
old inner ~ resp x 384 x old, each with the census coefficient of that layer. Every3 joins (0-2 at L7, 3-5 at L10, 6-8 at L13) are
kept; their own join cost is left unchanged (slightly conservative). Also prints the projected adjusted ratio for a raw ratio given on
the command line: raw x max(0.1, 0.1721 b) / 0.1721 (the lead's S11 util).
usage: births_bill.py <keep> <L0> [raw_ratio]"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]
keep, L0 = float(sys.argv[1]), int(sys.argv[2]); raw = float(sys.argv[3]) if len(sys.argv) > 3 else None
cen = json.loads((P2 / 'fable_analyst_v1/bill_by_layer_candS2_e2v7_k31r_bfix01_every3_r35.json').read_text()); pf = cen['per_layer_fn']
cost = lambda l, f: pf.get(f'{l}|{f}', 0)
s2 = next(x for x in (json.loads(y) for y in open(HERE.parent / 'leadergap/screen2.jsonl')) if x['rank'] == 35 and x['cfg'] == 's2')['counts']
total = [s2[str(l)]['total'] for l in range(16)]
width = [total[k + 1] - total[k] for k in range(15)]                  # cohort k = the columns born at layer k
joined_at = {k: 7 for k in (0, 1, 2)} | {k: 10 for k in (3, 4, 5)} | {k: 13 for k in (6, 7, 8)}
S2_all, new_all = 0.0, 0.0
for l in range(16):
    layer = sum(v for kk, v in pf.items() if kk.split('|')[0] == str(l)); S2_all += layer
    c = s2[str(l)]; prev, resp, live = c['prev'], c.get('count1', 1024), c.get('count0', 1024)
    yS2, oS2 = c['total'] - c['old'], c['old']
    y_push = y_read = o_new = 0.0
    for k in range(l):                                                # cohorts pushed at layer l
        w = width[k]; pruned = k >= L0
        is_old = k in joined_at and joined_at[k] <= l
        w_after = keep * w if pruned else w
        if is_old:
            o_new += w_after
        else:
            y_push += (w if (not pruned or l == k + 1) else w_after); y_read += w_after
    d = 0.0
    if yS2:
        d += cost(l, '_push_bank_shared_newborn') * (1 - y_push / yS2) + cost(l, '_shared_old384_read') * (1 - y_read / yS2)
    if oS2:
        d += (cost(l, '_slr_reconstruct') + cost(l, '_slr_inner')) * (1 - o_new / oS2)
    new_all += layer - d
b = new_all / S2_all
out = dict(keep=keep, L0=L0, whole_card_bill_ratio=round(b, 4), bill_change_pct=round(100 * (b - 1), 1))
if raw is not None:
    out.update(raw_ratio=raw, projected_adjusted_ratio=round(raw * max(0.1, 0.1721 * b) / 0.1721, 4))
print(json.dumps(out))
