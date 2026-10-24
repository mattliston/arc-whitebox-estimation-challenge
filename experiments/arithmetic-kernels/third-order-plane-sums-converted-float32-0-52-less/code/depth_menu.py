#!/usr/bin/env python3
"""TAG fable-engineer (E-28): depth TRADE MENU from the fitted compute model (depth_plan_v2.py on the bench) and a product census.
For every product family and depth step away from the census's current depths: products per net, bill change per net (Gflop and %
of the net's bill) and compute change per net (model seconds = in-process numpy, 1 thread, loaded box), and the exchange rate in
Gflop of bill per compute-second. Padded D6 (push64 style: k, n zero-padded to multiples of 64, plus the m*k + k*n pad copies) is
priced for the five-level pushes. usage: depth_menu.py <depth_bench.json> <product_profile.json (baseline census)>"""
import sys, json, runpy, io, contextlib, collections
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys_argv = sys.argv[:]
sys.argv = ["depth_plan_v2.py", sys_argv[1], sys_argv[2]]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(HERE / "depth_plan_v2.py"))
t_model, bill_model = g["t_model"], g["bill_model"]
prof = json.load(open(sys_argv[2]))
def family(kind, m, k, n, dep, path):
    if "w5" in path: return None
    if kind == "push":
        if m == 1024 and k == 384: return "push k=384 (slr reconstruct)"
        if m != 1024: return "push m=384"
        return "push k>=768, dims/64" if (k % 64 == 0 and n % 64 == 0) else "push k>=768, not /64"
    if kind == "sq":
        return "square 1024^3" if m == k == n == 1024 else "square other"
    if kind == "read":
        return "read n=384" if n == 384 else "read wide"
    return None
def steps(kind, m, k, n, dep):
    out = []
    if kind == "push":
        if dep == 6: out += [("D6->D5", 5, None), ("D6->D4", 4, None)]
        if dep == 5:
            out += [("D5->D4", 4, None)]
            if m == 1024 and k % 64 == 0 and n % 64 == 0: out += [("D5->D6", 6, None)]
            elif m == 1024: out += [("D5->D6 padded", 6, ((k + 63) // 64 * 64, (n + 63) // 64 * 64))]
        if dep == 4: out += [("D4->D5", 5, None)]
    elif kind == "sq":
        if dep == 5:
            out += [("D5->D4", 4, None)]
            if m % 64 == k % 64 == n % 64 == 0: out += [("D5->D6", 6, None)]
    elif kind == "read" and dep == 5:
        out += [("D5->D4", 4, None)]
    return out
agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
nets = len(prof)
for rec in prof:
    bill_tot = rec["util"] * 2 ** 41
    for key, cnt, s, gm, b, ops in rec["products"]:
        kind, m, k, n, dep, path = key
        if not isinstance(dep, int): continue
        fam = family(kind, m, k, n, dep, path)
        if fam is None: continue
        for name, L, pad in steps(kind, m, k, n, dep):
            if (kind, L) not in g["models"]: continue
            if pad:
                kp, npd = pad
                db = bill_model(kind, m, kp, npd, L) + m * k + k * n - bill_model(kind, m, k, n, dep)
                dt = t_model(kind, m, kp, npd, L) + (m * k + k * n) / 2e9 - t_model(kind, m, k, n, dep)
            else:
                db = bill_model(kind, m, k, n, L) - bill_model(kind, m, k, n, dep)
                dt = t_model(kind, m, k, n, L) - t_model(kind, m, k, n, dep)
            a = agg[(fam, name)]; a[0] += cnt; a[1] += cnt * db / 1e9; a[2] += cnt * dt
bill_mean = sum(r["util"] for r in prof) / nets * 2 ** 41 / 1e9
rows = []
for (fam, name), (cnt, dbg, dts) in agg.items():
    rows.append((fam, name, cnt / nets, dbg / nets, dbg / nets / bill_mean * 100, dts / nets, (-dbg / dts) if dts else float("inf")))
print(f"baseline census: {sys_argv[2]} ({nets} nets, mean bill {bill_mean:.1f} Gflop)")
print("BUY BILL (compute up), best rate first: family | step | products/net | bill change Gflop (% of bill) | compute s | Gflop per s")
for r in sorted([r for r in rows if r[5] > 0], key=lambda r: -r[6]):
    print(f"  {r[0]:26s} {r[1]:15s} {r[2]:5.1f}  {r[3]:+7.2f} ({r[4]:+6.2f}%)  {r[5]:+6.2f} s  {r[6]:6.2f}")
print("FREE COMPUTE (bill up), cheapest first: family | step | products/net | bill change | compute s | Gflop per s freed")
for r in sorted([r for r in rows if r[5] < 0], key=lambda r: r[6]):
    print(f"  {r[0]:26s} {r[1]:15s} {r[2]:5.1f}  {r[3]:+7.2f} ({r[4]:+6.2f}%)  {r[5]:+6.2f} s  {r[6]:6.2f}")
