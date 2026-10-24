#!/usr/bin/env python3
"""TAG fable-engineer (E-72c, 2026-09-25): paper price of the lead's k-split question. S23/S24's depth-5 bank read/push products with
inner dim >= 4096 (e.g. 928 x 8128 x 992) run at depth 5 because a depth-6 tree needs dims divisible by 64 and does not fit the arena.
Price: split k into chunks that fit S24's depth-6 regions (largest 64-multiple <= CAP; CAP = 4096 per the lead), pad m and n to
64-multiples, run each chunk at depth 6, add the chunk outputs ((nch-1) m'n' adds). Pad/unpad copies are NOT billed here.
Bill model: 2 (7/8)^L mkn + c_L (mk + kn + mn); c_5 and c_6 least-squares fitted on the measured bills in e72_pad6_bench.log (depth 5,
4 products) and e72_depth7_bench.log (depth 6, 6 products). CPU: EXTRAPOLATED from those logs' backend seconds per billed GFLOP
(depth 5 = mean of the 4 big-k products; depth 6 = band over the measured depth-6 shapes), not measured on chunks.
usage: python3 e72c_ksplit_price.py [CAP]   (census: request_census/product_classes_S23_r100.json = net 100, per net)"""
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
REG6 = (90354432, 120472576, 120472576)            # S24 _IPT_REGION[6] (A, B, C) floats
NET_BILL = 310.4e9                                 # S23 net 100 flops (RESULT.md E-71 census)
up = lambda x, q: -(-x // q) * q
meas = {5: [], 6: []}                              # (m, k, n, flops, backend_s)
for line in open(H / 'e72_pad6_bench.log'):
    r = json.loads(line)
    if 'L5' in r: meas[5].append((*r['dims'], r['L5']['flops'], r['L5']['backend_s']))
for line in open(H / 'e72_depth7_bench.log'):
    if not line.startswith('{'): continue
    r = json.loads(line)
    if isinstance(r.get('L6'), dict): meas[6].append((r['m'], r['k'], r['n'], r['L6']['flops'], r['L6']['backend_s']))
coef = {}
for L, rows in meas.items():
    S = [m * k + k * n + m * n for m, k, n, f, t in rows]
    R = [f - 2 * (7 / 8) ** L * m * k * n for m, k, n, f, t in rows]
    coef[L] = sum(r * s for r, s in zip(R, S)) / sum(s * s for s in S)
    worst = max(abs(2 * (7 / 8) ** L * m * k * n + coef[L] * s - f) / f for (m, k, n, f, t), s in zip(rows, S))
    print('depth %d: fitted transform coefficient %.2f per element of (mk+kn+mn); worst bill misfit %.2f%% on %d measured products' % (L, coef[L], 100 * worst, len(rows)))
bill = lambda L, m, k, n: 2 * (7 / 8) ** L * m * k * n + coef[L] * (m * k + k * n + m * n)
rate5 = sum(t / f for m, k, n, f, t in meas[5]) / len(meas[5]) * 1e9
rates6 = sorted(t / f * 1e9 for m, k, n, f, t in meas[6])
print('backend seconds per billed GFLOP: depth 5 big-k %.4f; depth 6 measured shapes %s' % (rate5, ' '.join('%.4f' % x for x in rates6)))
C = json.load(open(H / 'request_census' / 'product_classes_S23_r100.json'))
rows = []
for (m, k, n, L), c in C:
    if L != 5 or k < 4096: continue
    M, N = up(m, 64), up(n, 64)
    ck = min(CAP, 64 * min(REG6[0] // (7 ** 6 * (M // 64)), REG6[1] // (7 ** 6 * (N // 64))), up(k, 64))
    nch = -(-k // ck)
    chunks = [ck] * (nch - 1) + [up(k - ck * (nch - 1), 64)]
    for q in chunks:
        assert 7 ** 6 * (M // 64) * (q // 64) <= REG6[0] and 7 ** 6 * (q // 64) * (N // 64) <= REG6[1] and 7 ** 6 * (M // 64) * (N // 64) <= REG6[2]
    b5 = bill(5, m, k, n)
    b6 = sum(bill(6, M, q, N) for q in chunks) + (nch - 1) * M * N
    rows.append(dict(m=m, k=k, n=n, cnt=c, ck=ck, nch=nch, b5=b5 * c, b6=b6 * c, req5=90 * c, req6=(105 * nch + nch - 1) * c))
def summary(label, sel):
    b5 = sum(r['b5'] for r in sel); b6 = sum(r['b6'] for r in sel); dq = sum(r['req6'] - r['req5'] for r in sel)
    t5 = b5 / 1e9 * rate5
    print('%-34s %2d products/net  bill %5.1f GF = %4.1f%% of net bill | split %+.2f GF = %+.2f%% of net bill | requests %+5d | '
          'backend %.2f s -> %.2f..%.2f s (%+.1f..%+.1f s/net, extrapolated)' % (label, sum(r['cnt'] for r in sel), b5 / 1e9, 100 * b5 / NET_BILL,
          (b6 - b5) / 1e9, 100 * (b6 - b5) / NET_BILL, dq, t5, b6 / 1e9 * rates6[0], b6 / 1e9 * rates6[2], b6 / 1e9 * rates6[0] - t5, b6 / 1e9 * rates6[2] - t5))
    return b6 - b5
print('chunk cap %d; per class (m k n count chunk nch: split vs depth 5 on that product)' % CAP)
for r in sorted(rows, key=lambda r: r['b6'] - r['b5']):
    print('  %4d %5d %4d x%d  ck %4d nch %d  %+6.2f%%' % (r['m'], r['k'], r['n'], r['cnt'], r['ck'], r['nch'], 100 * (r['b6'] / r['b5'] - 1)))
summary('all depth-5 products, k >= 4096', rows)
summary("lead's 6-8k inner dims (k >= 5900)", [r for r in rows if r['k'] >= 5900])
summary('bill-negative subset', [r for r in rows if r['b6'] < r['b5']])
