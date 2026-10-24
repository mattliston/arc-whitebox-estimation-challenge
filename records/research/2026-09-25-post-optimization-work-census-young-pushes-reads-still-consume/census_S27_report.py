#!/usr/bin/env python3
"""TAG fable-engineer2 (lead 09:3x): S27 (9ff80a12) bill census by component on fable-porter's S21 layout (census_S21_report.py PT-38b), with the
callee lines MAPPED to S27 by source text (the live loop 3192-3415 is identical to S21's; _shared_old384_push grew 10 lines; _flms_cov_star is
re-emitted without the batched s1 pair, so no covariance/star half-split), plus a KERNEL-ROLE split from the innermost frame and op name
(bill_census_layers_e2.py rows): leaf matmul / transform add-sub / copy-cast-view / lazy bind / physics op. Also the S21 -> S27 deltas per
component (census_S21.json: no setup, so compare net 100). 1 U = 2*1024^3 flops. usage: census_S27_report.py <raw S27.json> <census_S21.json> <out.json>"""
import json, sys, collections
raw = json.load(open(sys.argv[1])); s21 = json.load(open(sys.argv[2])); U = raw['U']
KIND = {'push_young': 'per-birth', 'push_newborn': 'per-birth', 'push_old_basis': 'per-birth', 'old_reconstruct': 'per-birth', 'join': 'per-birth',
        'push_assembly': 'per-birth', 'read_young': 'per-birth', 'read_old': 'per-birth', 'read_prep': 'per-birth', 'omitted_radial': 'per-birth',
        'column_bookkeeping': 'per-birth', 'covariance': 'fixed', 'star_lowrank': 'fixed', 'planes_K21_K22': 'fixed', 'pair_map': 'fixed',
        'gn_dcov': 'fixed', 'edge': 'fixed', 'gnn': 'fixed', 'births_planes': 'fixed', 'star_formation': 'fixed', 'final_map': 'fixed', 'misc': 'fixed'}
JOIN_LINES = set(range(5084, 5141))          # S21 5084-5130
READ_PREP = {3271, 3272, 3273, 3276, 3278, 3279, 3280, 3282}
BOOK = {3386, 3389, 3393, 3397, 3398, 3380}
BIRTHS = set(range(3334, 3378)) - {3341}
COV_LINES = {6811, 6818, 6820, 6821}         # S27 _flms_cov_star: C32 cast, C32 @ w, _sym_upper, cast back (S21 6570/6579/6580 + the batched pair's C half)
def cat(r):
    ln, c1, c2, c3 = r['line'], r['callee1'], r['callee2'], r['callee3']
    if ln in (3267, 3234) and (c2.startswith('_shared_old384_push') or c1.startswith('_shared_old384_push')):
        sp = c2 if c2.startswith('_shared_old384_push') else c1; x = int(sp.split(':')[1]); inner = c3 if sp == c2 else c2
        if x in (5074, 5152):                # S21 5074, 5142
            if inner.startswith('_push_bank_shared_newborn'):
                y = int(inner.split(':')[1]); return 'push_young' if y in (5053, 5055) else ('push_newborn' if y in (5056, 5058, 5059) else 'push_assembly')
            return 'push_young'
        if x in JOIN_LINES: return 'join'
        if x == 5142: return 'push_old_basis'   # S21 5132
        if x == 5143: return 'old_reconstruct'  # S21 5133
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
        return 'covariance' if x in COV_LINES else 'star_lowrank'
    if ln in (3287, 3288, 3291, 3292, 3293, 3294, 3296): return 'planes_K21_K22'
    if ln == 3303: return 'pair_map'
    if ln == 3305: return 'gn_dcov'
    if ln == 3318: return 'edge'
    if ln in (3320, 3324, 3325, 3327, 3328): return 'gnn'
    if ln in BIRTHS: return 'births_planes'
    if ln in (3401, 3402): return 'star_formation'
    if ln in (3235, 3236, 3237): return 'final_map'
    return 'misc'
KERNEL = ('_ipt_', '_rs_', '_bb_', '_lvf', '_ks_', '_packed', '_square_product', '_mm', '_push6', '_read6', '_cp_d6', '_w5_port', '_sym_upper',
          '_shared_old384_mm', '_join_rows_mm', '_tr_', '_TR', '_D6X', '_m64')
BIND = ('_ipt_side', '_rs_level1', '_rs_basis', '_ipt_view', '_cp3_bind', '_bind_push', '_pb_', '_lvf_rs_level1', '_ks_bind', '_ks_leaf', '_e2_pair_arena')
COPYOPS = {'copyto', 'concatenate', 'stack', 'astype', 'reshape', 'copy', 'take', 'put', 'transpose', 'moveaxis', 'unstack', 'split', '__getitem__', 'ascontiguousarray', 'pad', 'zeros', 'empty', 'broadcast_to'}
def role(r):
    fn = r['inner'].split(':')[0] if r.get('inner') else ''
    if fn.startswith(BIND): return 'bind'
    if r['op'].split('|')[0] in ('matmul', 'dot', 'einsum', 'tensordot', 'vdot', 'inner', 'outer') and fn.startswith(KERNEL): return 'leaf_matmul' if r['op'].split('|')[0] != 'outer' else 'physics_op'
    if r['op'].split('|')[0] in ('matmul', 'dot', 'tensordot'): return 'leaf_matmul'
    if fn.startswith(KERNEL): return 'transform_addsub' if r['op'] in ('add', 'subtract', 'negative') else ('copy' if r['op'] in COPYOPS else 'kernel_other')
    return 'physics_op'
out = dict(tag=__doc__.split('\n')[0], cand=raw['cand'], U_flops=U, nets={})
for net in raw['ranks']:
    rows = [r for r in raw['rows'] if r['net'] == net]; tot = raw['totals'][str(net)]
    comp = collections.Counter(); roles = collections.Counter(); comp_role = collections.defaultdict(collections.Counter); per_layer = collections.defaultdict(collections.Counter)
    for r in rows:
        c = cat(r); ro = role(r); comp[c] += r['flops']; roles[ro] += r['flops']; comp_role[c][ro] += r['flops']
        per_layer[r['layer'] if r['layer'] is not None else -1][c] += r['flops']
    l15 = sum(per_layer[15].values())
    s = s21['nets'].get(str(net), {}).get('totals', {})
    out['nets'][str(net)] = dict(flops=tot, util=tot / 2 ** 41, U_total=round(tot / U, 2),
        components={k: dict(U=round(v / U, 3), pct=round(100 * v / tot, 3), kind=KIND[k], S21_U=s.get(k, {}).get('U'),
                             dU_vs_S21=(round(v / U - s[k]['U'], 3) if k in s else None), roles={x: round(y / U, 3) for x, y in comp_role[k].most_common() if y}) for k, v in comp.most_common()},
        roles={k: dict(U=round(v / U, 3), pct=round(100 * v / tot, 3)) for k, v in roles.most_common()},
        L15_bank_pass=dict(U=round(l15 / U, 3), pct=round(100 * l15 / tot, 3)),
        per_layer={str(l): {k: round(v / U, 3) for k, v in c.most_common() if v} for l, c in sorted(per_layer.items())})
json.dump(out, open(sys.argv[3], 'w'), indent=1)
for net, o in out['nets'].items():
    print(f"net {net}: {o['U_total']} U ({o['flops']:,} flops, util {o['util']:.5f}); S21 {s21['nets'].get(net, {}).get('U_total')} U")
    for k, v in o['components'].items():
        d = f"{v['dU_vs_S21']:+7.2f}" if v['dU_vs_S21'] is not None else '    new'
        print(f"   {v['kind']:9s} {k:18s} {v['U']:7.2f} U {v['pct']:6.2f}% | S21 {v['S21_U'] if v['S21_U'] is not None else '-':>7} dU {d} | {dict(list(v['roles'].items())[:3])}")
    print("   roles:", {k: (v['U'], v['pct']) for k, v in o['roles'].items()}, "| L15 bank pass:", o['L15_bank_pass'])
