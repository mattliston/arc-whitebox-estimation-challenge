#!/usr/bin/env python3
"""TAG fable-physicist (RF6-5): native bill ESTIMATE of the fold<L>:<r0>/<r> old-tier design against S2/S6b (every3, rank 384), in dense
1024^3 product units per layer (2 x 1024^3 flops). Calibration: the analyst's native per-layer, per-function census of S2
(fable_analyst_v1/bill_by_layer_candS2_e2v7_k31r_bfix01_every3_r35.json) and the torch S2 column counts on r35 (leadergap/screen2.jsonl).
Per layer, the census cost of each function is divided by its S2 feature to give a per-unit coefficient, then applied to the fold
features:
  young push       prev x young columns               -> 0 at layers >= L (no young columns)
  atoms read       resp x live x young columns        -> 0 at layers >= L
  old reconstruct  1024 x rank x old columns          -> 1024 x r x ALL columns  (the per-column cost scales with the frame rank)
  old inner        resp x rank x old columns          -> resp x r x ALL columns
  old push         the frame push U = W[prev]^T Q only, 0.6 x 2 x 1024 x prev x r flops (S2's every3 join costs are removed with the S2 push)
  joins            + a range finder + QR + projections at EVERY layer >= L, plain flops x 0.6 (Winograd), with the newborn width w:
                   2 [2 prev w r + 5 prev r^2 + r prev w + r^2 w + 2 r^3]
  lazy rotation    + one U @ map per layer (2 x 1024 x r^2 flops): the native lazy form absorbs the old-coordinate rotation
Everything else (covariance, star, pair map, learned parts, L15) is unchanged.
usage: fold_bill.py <L> <r0> <r>"""
import sys, json, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]
Lf, r0, r = (int(x) for x in sys.argv[1:4])
cen = json.loads((P2 / 'fable_analyst_v1/bill_by_layer_candS2_e2v7_k31r_bfix01_every3_r35.json').read_text()); pf = cen['per_layer_fn']
G = 2 * 1024 ** 3
cost = lambda l, f: pf.get(f'{l}|{f}', 0) / G
rows = [json.loads(x) for x in open(HERE.parent / 'leadergap/screen2.jsonl')]
s2 = next(x for x in rows if x['rank'] == 35 and x['cfg'] == 's2')['counts']
tot_s2, tot_fold = 0.0, 0.0; lines = []
for l in range(16):
    c = s2[str(l)]; layer_all = sum(v for k, v in pf.items() if k.split('|')[0] == str(l)) / G
    tot_s2 += layer_all
    if l < Lf:
        tot_fold += layer_all; continue
    young, old, allc, prev = c['total'] - c['old'], c['old'], c['total'], c['prev']
    resp, live = c.get('count1', 1024), c.get('count0', 1024)
    yp, yr = cost(l, '_push_bank_shared_newborn'), cost(l, '_shared_old384_read')
    rc, inn, op = cost(l, '_slr_reconstruct'), cost(l, '_slr_inner'), cost(l, '_shared_old384_push')
    jf = cost(l, '_slr_join_factors')
    rows_rc = 1024
    k_rc = rc / (rows_rc * 384 * old) if old else 0; k_in = inn / (resp * 384 * old) if old else 0
    new_rc = k_rc * rows_rc * r * allc; new_in = k_in * resp * r * allc
    w = (c['total'] - (s2[str(l - 1)]['total'] if l > 0 else 0)) if l < 15 else 0     # newborn width pushed at this layer
    w = max(w, 1088)
    joins = 0.6 * 2 * (2 * prev * w * r + 5 * prev * r * r + r * prev * w + r * r * w + 2 * r ** 3) / G
    lazy = 2 * 1024 * r * r / G
    new_op = 0.6 * 2 * 1024 * prev * r / G          # the frame push U = W[prev]^T Q only (joins are counted separately; x0.6 Winograd)
    fold_layer = layer_all - yp - yr - rc - inn - op - jf + new_rc + new_in + new_op + joins + lazy
    tot_fold += fold_layer
    lines.append(dict(layer=l, S2=round(layer_all, 2), fold=round(fold_layer, 2), young_cols=young, all_cols=allc,
                      parts=dict(removed_young=round(yp + yr, 2), old_rebuild_read=round(new_rc + new_in, 2), push=round(new_op, 2), joins=round(joins + lazy, 2))))
for x in lines: print(json.dumps(x))
print(json.dumps(dict(L=Lf, r0=r0, r=r, S2_per_layer=round(tot_s2 / 16, 2), fold_per_layer=round(tot_fold / 16, 2),
                      bill_change=round(100 * (tot_fold / tot_s2 - 1), 1),
                      layers_ge_L_S2=round(sum(x['S2'] for x in lines) / len(lines), 2), layers_ge_L_fold=round(sum(x['fold'] for x in lines) / len(lines), 2))))
