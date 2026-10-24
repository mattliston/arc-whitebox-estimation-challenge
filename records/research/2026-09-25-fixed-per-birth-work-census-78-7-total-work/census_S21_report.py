#!/usr/bin/env python3
"""TAG fable-porter (PT-38b): categorize bill_census_layers.py rows of the fired S21 (eddbcfcd) into PER-BIRTH (column-scaled) and FIXED
per-layer categories, in U (1 U = 2*1024^3 flops = one dense 1024^3 product) per layer and as % of the net's bill, with column counts.
Rules (live loop line numbers and callees of the S21 file): see CATS below. usage: census_S21_report.py <raw.json> <out census_S21.json>"""
import json, sys, collections
raw = json.load(open(sys.argv[1])); U = raw['U']
CATS = {  # name: (kind, description)
 'push_young':      ('per-birth', 'young CP columns pushed through W: _push_bank_shared_newborn inherited = W^T p_young (n x live x young)'),
 'push_newborn':    ('per-birth', "previous layer's births pushed: common = W^T base[:, selected] (n x live x births) + column assembly"),
 'push_old_basis':  ('per-birth', 'old tier: its rank-384 basis pushed, U = W^T Q (n x live x 384; fixed size once the tier exists)'),
 'old_reconstruct': ('per-birth', 'old tier: all old columns rebuilt explicitly, old = U F (n x 384 x old_cols)'),
 'join':            ('per-birth', 'old-tier join/recompress at L8/L11/L14 (sketch Y, QR, factors, weighted cores)'),
 'push_assembly':   ('per-birth', 'push bookkeeping: deadcut row gather/scatter, old|young concatenate, Qp gather'),
 'read_young':      ('per-birth', 'young read: (pp^2)[resp rows] @ (right*lam)^T over young columns (n_resp x young x n_live)'),
 'read_old':        ('per-birth', 'old read: inner = (pp^2)_old @ (F lam)^T (x old_cols x 384) + old plane = inner @ pre_basis^T'),
 'read_prep':       ('per-birth', 'read-side elementwise over columns: pp^2, pp^3*lam, row sums, gathers, plane put; L15 bank diagonal cube'),
 'omitted_radial':  ('per-birth', 'omitted_radial_source(pp, lam, ...)'),
 'column_bookkeeping': ('per-birth', 'newborn tuple gather, gated young columns p, old-tier commit, lam concatenate'),
 'covariance':      ('fixed', 'W^T C W: L0 Gram, half of the batched pair (C W), _sym_upper second product, L15 variance diagonal'),
 'star_lowrank':    ('fixed', 'star transport, low-rank: _lr_factor (sketch, 1 power step, small SVD), P1/pushed/s2 skinny products, half of the batched pair (pw W), L15 star diagonal'),
 'planes_K21_K22':  ('fixed', 'K21 plane assembly (cp + star + outer terms), k4, VV, CC = Cpre^2, K22 plane'),
 'pair_map':        ('fixed', 'the pair map (_nonlinear_pair_batch / terminal)'),
 'gn_dcov':         ('fixed', 'GN covariance correction _gn_dcov (L2-14)'),
 'edge':            ('fixed', 'message edge source (covariance_hook, L8-14)'),
 'gnn':             ('fixed', 'GNN mean head (hook) + recentering C -= m dm^T ...'),
 'births_planes':   ('fixed', 'birth planes (n^2): inherited, bp, trace_post, born, _bd_update (4 legs), cp_post'),
 'star_formation':  ('fixed', 'rest = target - cp_post - trace_post; star = 3 rest^T - 2 diag'),
 'final_map':       ('fixed', 'L15 scalar map + final GNN hook'),
 'misc':            ('fixed', 'everything else (mu, bpre, gates, selects, live outer, casts, W5 port begin/end)'),
}
JOIN_LINES = set(range(5084, 5131))
READ_PREP = {3271, 3272, 3273, 3276, 3278, 3279, 3280, 3282}
BOOK = {3386, 3389, 3393, 3397, 3398, 3380}
BIRTHS = set(range(3334, 3378)) - {3341}
def cat(r):
    ln, c1, c2, c3 = r['line'], r['callee1'], r['callee2'], r['callee3']
    if ln in (3267, 3234) and (c2.startswith('_shared_old384_push') or c1.startswith('_shared_old384_push')):
        sp = c2 if c2.startswith('_shared_old384_push') else c1; x = int(sp.split(':')[1]); inner = c3 if sp == c2 else c2
        if x in (5074, 5142):
            if inner.startswith('_push_bank_shared_newborn'):
                y = int(inner.split(':')[1]); return 'push_young' if y in (5053, 5055) else ('push_newborn' if y in (5056, 5058, 5059) else 'push_assembly')
            return 'push_young'
        if x in JOIN_LINES: return 'join'
        if x == 5132: return 'push_old_basis'
        if x == 5133: return 'old_reconstruct'
        return 'push_assembly'
    if ln == 3267: return 'push_assembly'
    if ln == 3274:
        x = int(c1.split(':')[1]) if c1 else 0
        return 'read_old' if x in (5015, 5016) else 'read_young'
    if ln == 3234:
        if c1.startswith('_l15_rowcut_k3:3908'): return 'star_lowrank'
        return 'read_prep'
    if ln in READ_PREP: return 'read_prep'
    if ln == 3310: return 'omitted_radial'
    if ln in BOOK: return 'column_bookkeeping'
    if ln in (3242, 3230): return 'covariance'
    if ln == 3246:
        x = int(c1.split(':')[1]) if c1 else 0
        if x == 6577: return 'HALF'
        return 'covariance' if x in (6570, 6579, 6580) else 'star_lowrank'
    if ln in (3287, 3288, 3291, 3292, 3293, 3294, 3296): return 'planes_K21_K22'
    if ln == 3303: return 'pair_map'
    if ln == 3305: return 'gn_dcov'
    if ln == 3318: return 'edge'
    if ln in (3320, 3324, 3325, 3327, 3328): return 'gnn'
    if ln in BIRTHS: return 'births_planes'
    if ln in (3401, 3402): return 'star_formation'
    if ln in (3235, 3236, 3237): return 'final_map'
    return 'misc'
agg = collections.defaultdict(lambda: collections.Counter())
for r in raw['rows']:
    c = cat(r); key = (r['net'], r['layer'] if r['layer'] is not None else -1)
    if c == 'HALF': agg[key]['covariance'] += r['flops'] / 2; agg[key]['star_lowrank'] += r['flops'] / 2
    else: agg[key][c] += r['flops']
snaps = {(s['net'], s['layer']): s for s in raw['snaps'] if s['layer'] is not None}
out = dict(tag='TAG fable-porter PT-38 S21 bill census (bill_census_layers.py + census_S21_report.py)', cand=raw['cand'],
           U_flops=U, note='net 35 cold (first in the process), net 100 warm (second); 1 U = 2*1024^3 flops = one dense 1024^3 product',
           categories={k: dict(kind=v[0], desc=v[1]) for k, v in CATS.items()}, nets={})
for net in raw['ranks']:
    tot = raw['totals'][str(net)]; layers = []
    for l in range(-1, 16):
        a = agg.get((net, l))
        if not a: continue
        s = snaps.get((net, l), {}); s1 = snaps.get((net, l + 1), {})
        cols, old = s.get('cols'), s.get('old_cols')
        births = ((s1.get('cols') - cols) // 2) if (s1.get('cols') is not None and cols is not None) else None
        layers.append(dict(layer=l, cols_in=cols, old_cols=old, young_cols=(cols - old) if cols is not None and old is not None else None,
                           old_frame_rank=(s.get('Q') or [None, None])[1], births=births, live_rows=s.get('live'),
                           U_total=round(sum(a.values()) / U, 3), U={k: round(v / U, 3) for k, v in sorted(a.items()) if v},
                           pct_of_bill={k: round(100 * v / tot, 3) for k, v in sorted(a.items()) if v}))
    T = collections.Counter()
    for k2, a in agg.items():
        if k2[0] == net: T.update(a)
    per_birth = sum(v for k, v in T.items() if CATS[k][0] == 'per-birth'); fixed = sum(v for k, v in T.items() if CATS[k][0] == 'fixed')
    out['nets'][str(net)] = dict(flops=tot, util=tot / 2 ** 41, U_total=round(tot / U, 2), per_birth_U=round(per_birth / U, 2), fixed_U=round(fixed / U, 2),
                                 per_birth_pct=round(100 * per_birth / tot, 2), fixed_pct=round(100 * fixed / tot, 2),
                                 totals={k: dict(U=round(v / U, 3), pct=round(100 * v / tot, 3), kind=CATS[k][0]) for k, v in sorted(T.items(), key=lambda kv: -kv[1])},
                                 layers=layers)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
for net, o in out['nets'].items():
    print(f"net {net}: {o['U_total']} U ({o['flops']:,} flops, util {o['util']:.5f}); PER-BIRTH {o['per_birth_U']} U = {o['per_birth_pct']}%, FIXED {o['fixed_U']} U = {o['fixed_pct']}%")
    for k, v in o['totals'].items(): print(f"   {v['kind']:9s} {k:18s} {v['U']:8.2f} U {v['pct']:7.2f}%")
