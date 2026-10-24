"""TAG opus_engineer a30: what would 7-level half-zero strips be worth? Stored census only (census/s38_4net.json = the fired S38 file on
networks 35, 100, 36, 45; census2.py). For every tree product that runs at 6 levels today, the bill of the same product at 7 levels
is estimated with the parent's fitted bill model (_p6_bill) on the dimensions padded to multiples of 64, a dimension that is then an
odd multiple of 64 carried as a half-zero strip (4 of the 7 deepest chunk products on its last leaf row / column, 3 chunks without
the strip's inner index: the saving terms of hzs/fragment_gen_v7.py's plan at one level deeper), and 0.98 flops per copied element
for the packing. Calibration: the same model against the measured bill of the products that run at 7 levels today.
usage: price_hz7.py [<census.json>]"""
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

E = Path(__file__).resolve().parents[1]
c = json.load(open(sys.argv[1] if len(sys.argv) > 1 else E/'census/s38_4net.json'))
tot = {n['rank']: n['flops'] for n in c['nets']}
S = lambda L: ((7 / 4) ** L - 1) / 0.75


def packet_bill(L, M, K, w):          # one column packet beyond the shared A side (parent's fitted model)
    return 2 * M * K * w * (7 / 8) ** L - M * w * (7 / 4) ** L + 0.7668 * K * w * S(L) + 1.5213 * M * w * S(L) + 2.6307 * (K * w + M * w)


def push_bill(L, m, k, n, pk):
    """Packet push at L levels with strips: (bill, moved elements)."""
    U = 1 << L; H = U // 2
    M, K = (m + H - 1) // H * H, (k + H - 1) // H * H                 # padded to multiples of U / 2
    hzr, hzk = M % U == H, K % U == H
    M, K = (M + U - 1) // U * U, (K + U - 1) // U * U                 # the tree's dimensions
    N = (n + U - 1) // U * U
    packets = [min(N, j0 + pk) - j0 for j0 in range(0, N, pk)]
    bill = 0.7669 * M * K * S(L) + 2.6307 * M * K + sum(packet_bill(L, M, K, w) for w in packets)
    bill += 0.98 * ((M * K if (hzr or hzk or (M, K) != (m, k)) else 0) + (K * N if (hzk or K != k or N != n) else 0) + (m * n if (hzr or N != n) else 0))
    for w in packets:
        cbt = w // U
        ra = M // U - (1 if hzr else 0)
        if hzr:
            bill -= 3 * 7 ** (L - 1) * (2 * (K // U) - 1) * cbt
        if hzk:
            bill -= 3 * 7 ** (L - 1) * 2 * ra * cbt
    return bill


def depth_of(p):
    d = set()
    for key in p['leaf']:
        shape = ast.literal_eval(key)[0]
        d.add(len(shape) - 2)
    return tuple(sorted(d))


rows = []
for p in c['products']:
    name = next((x for x in p['chain'] if x not in ('_mm', '_mm_batch', '_ipt_product_ab', '_ks_product', '_hz_ks_product', '_hzg_product')), '?')
    sh = p['shapes']
    if len(sh) < 2 or len(sh[0]) != 2 or len(sh[1]) != 2 or not p['flops']:
        continue
    rows.append(dict(name=name, m=sh[0][0], k=sh[0][1], n=sh[1][1], depth=depth_of(p), flops=p['flops'], rank=p['rank'], layer=p['layer'], calls=p.get('ops', 1)))
nets = len(tot); T = sum(tot.values())
print('%d networks, %.1f Gflop per network; tree products %.1f%% of the bill' % (nets, T / nets / 1e9, 100 * sum(r['flops'] for r in rows) / T))
by = defaultdict(float)
for r in rows:
    by[(r['name'][:24], r['depth'])] += r['flops']
print('share of the bill by family and levels of recursion:')
for (name, depth), f in sorted(by.items(), key=lambda kv: -kv[1])[:16]:
    print('   %-26s levels %-8s %6.2f%%' % (name, depth, 100 * f / T))
# calibration on today's 7-level packet pushes
print('calibration, packet pushes that run at 7 levels today (model / measured):')
for r in rows:
    if r['name'] == '_deadcut_push' and r['depth'] == (7,) and r['n'] > 1024 and r['rank'] == 35:
        print('   m %4d k %4d n %5d: %.3f' % (r['m'], r['k'], r['n'], push_bill(7, r['m'], r['k'], r['n'], 2048) / r['flops']))
print('calibration, packet pushes at 6 levels today (model at 6 levels with strips, packets of 4096 / measured):')
for r in rows:
    if r['name'] == '_deadcut_push' and r['depth'] == (6,) and r['n'] > 1024 and r['rank'] == 35 and r['k'] >= 512:
        print('   m %4d k %4d n %5d: %.3f' % (r['m'], r['k'], r['n'], push_bill(6, r['m'], r['k'], r['n'], 4096) / r['flops']))
# the lever: 6-level packet pushes with m, k <= 1024 at 7 levels with strips
save = defaultdict(float); moved = 0.0; cand = 0.0
detail = []
for r in rows:
    if r['name'] == '_deadcut_push' and r['depth'] == (6,) and r['n'] > 1024 and r['m'] <= 1024 and r['k'] <= 1024:
        b7 = push_bill(7, r['m'], r['k'], r['n'], 2048)
        cand += r['flops']
        gain = r['flops'] - b7
        if gain > 0:
            save['push'] += gain; moved += r['flops']
            detail.append((gain / T * 100, r))
cand_push = cand
print('6-level packet pushes: %.2f%% of the bill; the model moves %.2f%% of the bill to 7 levels and saves %.2f%% of the bill' % (100 * cand / T, 100 * moved / T, 100 * save['push'] / T))
agg = defaultdict(lambda: [0.0, 0.0, 0])
for g, r in detail:
    key = (r['m'], r['k'])
    agg[key][0] += g; agg[key][1] += r['flops'] / T * 100; agg[key][2] += 1
print('   by (m, k): saving %% of bill, share of bill, products')
for key, v in sorted(agg.items(), key=lambda kv: -kv[1][0])[:25]:
    print('   m %4d (mod 128: %3d) k %4d (mod 128: %3d): -%.3f%%  of %.2f%%  (%d products, %.1f%% each)' % (key[0], key[0] % 128, key[1], key[1] % 128, v[0], v[1], v[2], 100 * v[0] / v[1]))


# ---- plane reads (k-split: m x k x n with k >= 2048 cut into chunks that fit the tree regions)
def read_bill(L, m, k, n):
    U = 1 << L; H = U // 2
    M, N = (m + H - 1) // H * H, (n + H - 1) // H * H
    hzr, hzc = M % U == H, N % U == H
    M, N, K = (M + U - 1) // U * U, (N + U - 1) // U * U, (k + U - 1) // U * U
    region = {6: (90354432, 120472576), 7: (7 ** 7 * 128, 7 ** 7 * 128)}[L]   # 7 levels: the private regions of the parent's read kernel (_R7_REGION)
    cmax = U * min(region[0] // (7 ** L * (M // U)), region[1] // (7 ** L * (N // U)))
    if cmax < U:
        return None
    nb = K // U; nch = -(-nb // (cmax // U)); base, r = divmod(nb, nch)
    sizes = [(base + 1) * U] * r + [base * U] * (nch - r)
    CL, CC = 0.7669 * S(L) + 2.6307, 1.5213 * S(L) + 2.6307
    b = (sum(2 * (7 / 8) ** L * M * q * N + CL * (M * q + q * N) + (CC - (7 / 4) ** L) * M * N for q in sizes)   # per chunk: leaf products (their -r c term is in the last coefficient) and the two forward transforms
         + (nch - 1) * m * n + m * n - (nch - 1) * (CC * M * N + m * n - 7 ** L * (M // U) * (N // U)))
    b += (2.0 * m * k if (hzr or K != k or M != m) else 0) + (2.0 * k * n if (hzc or K != k or N != n) else 0) + 2.0 * m * n
    b -= (3 * 7 ** (L - 1) * 2 * (K // U) * (N // U) if hzr else 0) + (3 * 7 ** (L - 1) * 2 * (M // U - (1 if hzr else 0)) * (K // U) if hzc else 0)
    return b


print('calibration, plane reads (model / measured), network 35:')
seen = set()
for r in rows:
    if r['name'] == '_cp_read_young_sq' and r['rank'] == 35 and r['k'] >= 2048 and r['depth'] in ((6,), (7,)) and (r['m'], r['k'], r['n']) not in seen:
        seen.add((r['m'], r['k'], r['n']))
        b = read_bill(r['depth'][0], r['m'], r['k'], r['n'])
        if b:
            print('   levels %d: m %4d (mod 128: %3d) k %5d n %4d (mod 128: %3d): %.3f' % (r['depth'][0], r['m'], r['m'] % 128, r['k'], r['n'], r['n'] % 128, b / r['flops']))
cand = moved = saved = 0.0; agg = defaultdict(lambda: [0.0, 0.0, 0])
for r in rows:
    if r['name'] == '_cp_read_young_sq' and r['depth'] == (6,) and r['k'] >= 2048 and r['m'] <= 1024 and r['n'] <= 1024:
        cand += r['flops']
        b7 = read_bill(7, r['m'], r['k'], r['n'])
        b6 = read_bill(6, r['m'], r['k'], r['n'])
        if b7 and b6:
            gain = r['flops'] * (1 - b7 / b6)          # the model's ratio applied to the measured bill
            if gain > 0:
                moved += r['flops']; saved += gain
                key = (r['m'] % 128, r['n'] % 128)
                agg[key][0] += gain; agg[key][1] += r['flops']; agg[key][2] += 1
cand_read = cand
print('6-level plane reads: %.2f%% of the bill; the model moves %.2f%% of the bill to 7 levels and saves %.2f%% of the bill' % (100 * cand / T, 100 * moved / T, 100 * saved / T))
print('   by (m mod 128, n mod 128): saving % of bill, share of bill, products, saving per product')
for key, v in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    print('   m mod 128 = %3d, n mod 128 = %3d: -%.3f%%  of %.2f%%  (%d products, %.1f%% each)' % (key[0], key[1], 100 * v[0] / T, 100 * v[1] / T, v[2], 100 * v[0] / v[1]))


# ---- summary with a threshold on the saving per product, and the wall price by the release lane's platform rates
# (O/release/RESULT.md section 10: the 7-level packet pushes of today, 8.4% of the bill here, cost 5.5 s of wall; the 7-level reads, 3.9%, cost 2.1 s)
push7 = sum(r['flops'] for r in rows if r['name'] == '_deadcut_push' and r['depth'] == (7,) and r['n'] > 1024) / T
read7 = sum(r['flops'] for r in rows if r['name'] == '_cp_read_young_sq' and r['depth'] == (7,)) / T
print('today at 7 levels: packet pushes %.2f%% of the bill, plane reads %.2f%%' % (100 * push7, 100 * read7))
summary = []
for floor in (0.0, 2.0, 3.0, 4.0):
    out = {}
    for fam, name, bill_at_7, share7, secs7 in (('pushes', '_deadcut_push', None, push7, 5.5), ('reads', '_cp_read_young_sq', None, read7, 2.1)):
        moved = saved = 0.0
        for r in rows:
            if r['name'] != name or r['depth'] != (6,) or r['m'] > 1024:
                continue
            if fam == 'pushes' and r['n'] > 1024 and r['k'] <= 1024:
                gain = 1 - push_bill(7, r['m'], r['k'], r['n'], 2048) / r['flops']
            elif fam == 'reads' and r['k'] >= 2048 and r['n'] <= 1024 and read_bill(7, r['m'], r['k'], r['n']):
                gain = 1 - read_bill(7, r['m'], r['k'], r['n']) / read_bill(6, r['m'], r['k'], r['n'])
            else:
                continue
            if gain * 100 > floor:
                moved += r['flops'] / T; saved += gain * r['flops'] / T
        out[fam] = (saved, moved, secs7 * moved / share7)
    s = sum(v[0] for v in out.values()); w = sum(v[2] for v in out.values())
    print('products that save more than %.0f%%: bill -%.2f%% (pushes -%.2f%%, reads -%.2f%%), %.1f%% of the bill moved to 7 levels, wall about +%.1f s (%.0f s per 1%% of bill)'
          % (floor, 100 * s, 100 * out['pushes'][0], 100 * out['reads'][0], 100 * (out['pushes'][1] + out['reads'][1]), w, w / (100 * s) if s else 0))
    summary.append(dict(floor=floor, bill=-100 * s, pushes=-100 * out['pushes'][0], reads=-100 * out['reads'][0], moved=100 * (out['pushes'][1] + out['reads'][1]), wall_s=w))
(Path(__file__).resolve().parent/'price_summary.json').write_text(json.dumps(dict(census='census/s38_4net.json', networks=sorted(tot), six_level_pushes=100 * cand_push / T,
                                                                               six_level_reads=100 * cand_read / T, rows=summary), indent=1))
