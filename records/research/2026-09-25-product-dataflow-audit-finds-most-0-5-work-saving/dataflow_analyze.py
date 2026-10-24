#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-DFA, lever 1): analysis of dataflow_<tag>.json. (1) re-association of product chains (a product output consumed
by exactly one consumer, a product, through a plain or transposed view; slices noted), dense volume now vs re-associated, bill change at
0.50 / 0.53 / 0.56 of 2mkn; (2) roots feeding >= 2 products (shared operands), by side; (3) provenance of the operands of every product
with contraction length >= 2048 (Khatri-Rao check); (4) products by site, with operand provenance kinds.  usage: dataflow_analyze.py <tag>"""
import sys, json, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / f"dataflow_{sys.argv[1]}.json")); P, T, U, TOT = D["products"], D["tags"], D["uses"], D["total"]
def kind(t):
    if t is None: return "?"
    x = T[t]; return f"P{x['pid']}" if x["op"] == "product" else x["op"]
print(f"bill {TOT/1e9:.2f} G; products {len(P)} ({sum(p['bill'] for p in P)/1e9:.1f} G = {100*sum(p['bill'] for p in P)/TOT:.1f}%)")
# (4) products by site
bysite = collections.defaultdict(lambda: [0, 0, 0, collections.Counter()])
for p in P:
    s = bysite[(p["entry"], p["site"])]; s[0] += 1; s[1] += p["bill"]; s[2] += p["vol"] or 0
    if "a" in p: s[3][(kind(p["a"]).split("P")[0] or "product", kind(p["b"]).split("P")[0] or "product")] += 1
print("\n(4) products by entry and caller site (top 25 by bill): calls, bill %, bill/2mkn, operand provenance (A, B)")
for (e, st), (n, b, v, c) in sorted(bysite.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"   {e:22s} {st:34s} {n:4d}  {100*b/TOT:6.2f}%  {b/v if v else 0:5.3f}  {dict(c.most_common(3))}")
# (1) association
print("\n(1) single-use product -> product chains, re-associated")
rows = []
for p2 in P:
    if "a" not in p2 or p2["mkn"] is None: continue
    for side in ("a", "b"):
        t = p2[side]
        if t is None or T[t]["op"] != "product": continue
        cons = U.get(str(t), [])
        if len(cons) != 1: continue
        p1 = P[T[t]["pid"]]
        if "a" not in p1 or p1["mkn"] is None or p1.get("batch", 1) > 1 and p1["entry"] != "_mm_batch": continue
        m1, k1, n1 = p1["mkn"]; m2, k2, n2 = p2["mkn"]; v = p2["va" if side == "a" else "vb"]
        cur = 2 * m1 * k1 * n1 + 2 * m2 * k2 * n2
        if side == "a":   # P2 = X' @ C, X' is (m2 x k2), C is (k2 x n2)
            if v.startswith("N"): alt = 2 * k1 * m2 * n2 * 0 + 2 * k1 * n1 * n2 + 2 * m1 * k1 * n2   # A@(B@C)
            else: alt = 2 * k1 * m1 * n2 + 2 * n1 * k1 * n2                                         # B^T@(A^T@C)
        else:             # P2 = C @ X', C is (m2 x k2)
            if v.startswith("N"): alt = 2 * m2 * m1 * k1 + 2 * m2 * k1 * n1                          # (C@A)@B
            else: alt = 2 * m2 * n1 * k1 + 2 * m2 * k1 * m1                                         # (C@B^T)@A^T
        rows.append((alt - cur, p1, p2, side, v, cur, alt))
rows.sort(key=lambda r: r[0])
for d, p1, p2, side, v, cur, alt in rows[:20]:
    print(f"   P{p1['id']} {p1['entry']}@{p1['site']} {p1['mkn']} -> P{p2['id']} {p2['entry']}@{p2['site']} {p2['mkn']} side {side} view {v}: "
          f"dense {cur/1e9:.2f} -> {alt/1e9:.2f} G; bill {100*0.50*d/TOT:+.2f}% / {100*0.53*d/TOT:+.2f}% / {100*0.56*d/TOT:+.2f}%")
gain = sum(min(0, r[0]) for r in rows)
print(f"   chains found {len(rows)}; all cheaper re-associations together: dense {gain/1e9:+.2f} G = bill {100*0.53*gain/TOT:+.2f}% (0.53 x 2mkn)")
# (2) shared operands
print("\n(2) operand roots feeding >= 2 products (top 20 by the products' bill)")
share = collections.defaultdict(list)
for p in P:
    if "a" not in p: continue
    for side in ("a", "b"):
        if p[side] is not None: share[p[side]].append((p["id"], side, p["va" if side == "a" else "vb"], p["bill"]))
srows = sorted(((t, L) for t, L in share.items() if len(L) >= 2), key=lambda x: -sum(y[3] for y in x[1]))
for t, L in srows[:20]:
    x = T[t]; sides = collections.Counter(f"{s}{v}" for _, s, v, _ in L)
    print(f"   tag {t} {kind(t):14s} shape {x['shape']}: {len(L)} products, bill {100*sum(y[3] for y in L)/TOT:.2f}%, sides {dict(sides)}, "
          f"sites {sorted(set(P[i]['site'] for i, _, _, _ in L))[:4]}")
# (3) long-k provenance
print("\n(3) products with contraction length >= 2048: operand provenance (op <- parents, 3 levels)")
def chain(t, d=0):
    if t is None or d > 2: return ""
    x = T[t]; s = f"{kind(t)}{x['shape']}"
    if x.get("in_shapes"): s += f"(in {x['in_shapes'][:3]})"
    ps = [q for q in x.get("parents", []) if q is not None][:2]
    return s + ("" if not ps else " <- [" + "; ".join(chain(q, d + 1) for q in ps) + "]")
seen = set()
for p in sorted(P, key=lambda p: -p["bill"]):
    if "a" not in p or p["mkn"] is None or p["mkn"][1] < 2048: continue
    key = (p["site"], chain(p["a"])[:60], chain(p["b"])[:60])
    if key in seen: continue
    seen.add(key)
    print(f"   P{p['id']} {p['entry']}@{p['site']} mkn {p['mkn']} bill {100*p['bill']/TOT:.2f}%\n      A: {chain(p['a'])[:330]}\n      B: {chain(p['b'])[:330]}")
