"""TAG opus-release: how much bill the tree kernels could still give for more wall and requests (lead's pricing question of
2026-10-02 19:14Z: a public bank at the leaders' residual and wall). No session: the engineer's product census of one
network (O/engineer/census/<name>.json: every product with its shapes, its leaf patterns and its bill by operation) and
his fitted models of the alt-basis tree kernel (the candidate's _p6_bill / _p6_secs: bill beyond the leaf and backend
seconds as functions of the shapes and the depth L).
Prints: the bill by kind of operation and by plan (depth, packets, strips, upper blocks); how well the bill model reproduces
the measured bill of the single-plan products; and what ONE MORE LEVEL would do, family by family, wherever the model says
the bill falls (operands zero-padded to the next multiple, the padding billed). Arena limits and the per-leaf call cost of
numpy are not modelled: the printed gains are ceilings, the printed seconds are floors.
usage: bill_room.py [census json]      (default: O/engineer/census/s40_651.json)"""
import collections
import json
from pathlib import Path
import statistics as st
import sys

O = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else O/'engineer'/'census'/'s40_651.json'
d = json.loads(path.read_text()); P = d['products']; total = d['nets'][0]['flops']
REQ_PER_LEVEL = 18            # requests one more level adds to a product (_DR_REQ: 75 / 92 / 110 at depths 4 / 5 / 6)
PLATFORM_PER_LOCAL = 1.61     # platform kernel seconds per local kernel second (S40: 70.65 s against 43.9 s)


def bill(L, m, k, n, pad=0):
    S = ((7/4)**L - 1)/0.75
    return (2*m*k*n*(7/8)**L - m*n*(7/4)**L + 0.7669*m*k*S + 0.7668*k*n*S + 1.5213*m*n*S + 2.6307*(m*k + k*n + m*n) + 0.9768*pad)


def secs(L, m, k, n, pad=0):
    S = ((7/4)**L - 1)/0.75
    return (4.22e-4 + 1.18e-7*7**L + 9.45e-12*2*m*k*n*(7/8)**L + 4.63e-10*(m*k + k*n + m*n)*S + 1.36e-9*pad + 1.73e-10*m*n*(7/4)**L)


def plan(p):
    """(depth, family) of a product from its leaf patterns: a tree of depth L has shape (7,)*L + leaf; the strip kernels group
    the 7 products of the last level (a leading 1 or 2 in the pattern); the upper-block products batch four block products."""
    pats = [json.loads(s) for s in p['leaf']]
    batch = p['chain'][-1] == '_psdb_sym_upper'
    depth = max(len(a[0]) - 2 - (1 if (batch and a[0][0] != 7) else 0) for a in pats)
    strips = not batch and any(a[0][0] in (1, 2) and len(a[0]) == 8 for a in pats)
    family = ('upper-block symmetric products' if batch else 'strips at depth 6' if strips
              else f'depth {depth}' + (' in column or k packets' if len(pats) > 1 else ''))
    return depth, family, len(pats) == 1 and not batch


print(f"{path.name}: network {d['nets'][0]['rank']}, bill {total/1e9:.1f} Gflop, {len(P)} products")
inside = {k: sum(p[k] for p in P) for k in ('matmul', 'addsub', 'rest')}
print(f"bill by kind: leaf multiplications {100*inside['matmul']/total:.1f}%, tree additions {100*inside['addsub']/total:.1f}%, other work inside "
      f"products {100*inside['rest']/total:.1f}%, operations that are no product {100*(1 - sum(p['flops'] for p in P)/total):.1f}%")
rows, fam = [], collections.defaultdict(lambda: [0, 0])
for p in P:
    if not p['leaf'] or not p['flops']:
        continue
    (m, k), (_, n) = p['shapes']; depth, family, single = plan(p)
    fam[family][0] += 1; fam[family][1] += p['flops']
    rows.append(dict(p=p, m=m, k=k, n=n, L=depth, family=family, single=single))
print('bill by plan:')
for family, (count, flops) in sorted(fam.items(), key=lambda kv: -kv[1][1]):
    print(f"   {family:36s} {100*flops/total:6.2f}%  {count:3d} products")
single = [r for r in rows if r['single'] and r['L'] >= 3]
err = sorted(bill(r['L'], r['m'], r['k'], r['n'])/r['p']['flops'] - 1 for r in single)
print(f"bill model against the measured bill of {len(single)} single-plan products: median {100*st.median(err):+.2f}%, "
      f"10th-90th percentile {100*err[len(err)//10]:+.2f}..{100*err[-len(err)//10]:+.2f}%")
print('one more level wherever the model says the bill falls (bill change as a share of the network bill, requests, platform wall):')
out = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])
for r in rows:
    if r['family'].startswith('upper-block') or r['L'] == 0:
        continue
    m, k, n, L0 = r['m'], r['k'], r['n'], r['L']
    L = L0 + 1; u = 1 << L
    M, K, N = -(-m//u)*u, -(-k//u)*u, -(-n//u)*u
    pad = (M*K if (M, K) != (m, k) else 0) + (K*N if (K, N) != (k, n) else 0) + (m*n if (M, N) != (m, n) else 0)
    rel = bill(L, M, K, N, pad)/bill(L0, m, k, n) - 1
    key = 'below depth 6' if L0 < 6 else r['family']
    o = out[key]; o[0] += 1
    if rel < 0:
        o[1] += rel*r['p']['flops']; o[2] += secs(L, M, K, N, pad) - secs(L0, m, k, n); o[3] += 1
for key, (count, gain, seconds, moved) in sorted(out.items(), key=lambda kv: kv[1][1]):
    print(f"   {key:36s} {moved:3d} of {count:3d} products: {100*gain/total:+.3f}% of the bill, +{REQ_PER_LEVEL*moved:,} requests, "
          f"+{PLATFORM_PER_LOCAL*seconds:.1f} s")
plain = [p for p in P if p['leaf'] and p['flops'] and plan(p)[0] == 0]
print(f"products that run as plain matmuls: {len(plain)}, {100*sum(p['flops'] for p in plain)/total:.2f}% of the bill")
same = collections.defaultdict(lambda: [0, 0])
for p in P:
    if p['chain'][-2:] == ['_flms_cov_star', '_mm'] or p['chain'][-1] == '_psdb_sym_upper':
        same[p['chain'][-1]][0] += 1; same[p['chain'][-1]][1] += p['flops']
print('the covariance sandwich of every layer (the same bill on every network): ' + '; '.join(
    f"{name} {count} calls, {flops/1e9:.1f} Gflop, {100*flops/total:.2f}% of the bill" for name, (count, flops) in same.items()))
