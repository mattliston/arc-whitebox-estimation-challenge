"""TAG opus-trainer RAD (10-12 17:0xZ; lead 17:0xZ: "YES, radlab's aggregate weak-band arm for S53"): writes the next version of a round's make file
(S53 make_s53_v6.py, S54 make_s54_v4.py, S55-S66 make_s<N>_v4.py) with rule (d): ONE aggregate weak-band arm 'fagg', pre-declared here.
  - Candidates: every family and band rule (b) can see (the route table, the same 64 gradients and t), kappa out.
  - A band enters if t^2 >= 4 on those gradients, it is not a strict band of rule (b), and it does not overlap a strict band of its own family.
  - At most one band per family: the one with the largest |t|.
  - Each band is moved against its gradient sign by 0.02 x min(1, |t| / 6) (proportional to t, capped at the dial cap 0.02), all inside ONE arm
    file (start + those shifts), written as the fixed-arm dir arms_fixed_v1 (live ['fagg']; rule (c) is off, so the dir is otherwise empty).
  - No band -> no arm. It is sized like any arm (one column of the freeze; the two-stage prune applies to it).
usage (from O/trainer): python -B cert/RAD/make_agg_arm_v1.py <make in> <make out (new file)>"""
import os
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
assert not dst.exists(), dst
s = src.read_text()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("""    DF = HOME / 'arms_fixed_v1'
    if spec:
        subprocess.run([PY, '-B', RAD, 'fixed', str(HOME / 'config.json'), str(DF), ','.join(spec)], check=True)
    keep = read(DG / 'arms.json')['live'] + (read(DF / 'arms.json')['live'] if spec else [])
""", """    DF = HOME / 'arms_fixed_v1'
    if spec:
        subprocess.run([PY, '-B', RAD, 'fixed', str(HOME / 'config.json'), str(DF), ','.join(spec)], check=True)
    agg = {}      # (d) ONE aggregate weak-band arm 'fagg' (lead 17:0xZ, radlab; cert/RAD/make_agg_arm_v1.py)
    for fam in sorted(rows[0]['pl_grads'], key=str.lower):
        if routes[fam] is None or fam == 'kappa':
            continue
        G = np.stack([r['pl_grads'][fam].double().numpy() * (1e8 / r['scale']) for r in rows])
        if not np.abs(G).max() > 0:
            continue
        strict = [(0, 15) if ':' not in x else tuple(int(y) for y in x.split(':')[1:]) for x in dials if x.split(':')[0] == fam]
        best = None
        for lo, hi in (BANDS[fam] if routes[fam] == 'bands' else DEFAULT_BANDS if routes[fam] == 'layer' else [(0, 15)]):
            if any(not (hi < a or lo > b) for a, b in strict):
                continue
            tt = t(G[:, lo:hi + 1].sum(1))
            if tt * tt >= 4 and (best is None or abs(tt) > abs(best[2])):
                best = (lo, hi, tt)
        if best:
            agg[fam] = best
    if agg and not spec:
        sys.path.insert(0, '.')
        from otr_common import load_params, load_pl, save_params
        A0, d0 = load_params(torch, cfg['params']); pl = {k: [float(x) for x in v] for k, v in load_pl(torch, cfg['params']).items()}
        shifts = {}
        for fam, (lo, hi, tt) in sorted(agg.items()):
            step = float(-np.sign(tt) * 0.02 * min(1.0, abs(tt) / 6.0))
            pl[fam] = [x + (step if lo <= l <= hi else 0.0) for l, x in enumerate(pl.get(fam, [0.0] * 16))]
            shifts[fam] = dict(lo=lo, hi=hi, t=tt, shift=step)
        DF.mkdir(parents=True)
        save_params(torch, DF / 'arm_fagg.pt', A0, d0, pl); (DF / 'arm_fagg_hu.json').write_text(json.dumps(read(cfg['hu'])))
        (DF / 'arms.json').write_text(json.dumps(dict(live=['fagg'], size=dict(fagg=1.0), base=cfg['params'], base_hu=cfg['hu'], shifts=shifts,
                                                      spec='rule (d): one aggregate weak-band arm (make_agg_arm_v1.py)'), indent=1))
    keep = read(DG / 'arms.json')['live'] + (read(DF / 'arms.json')['live'] if (DF / 'arms.json').exists() else [])
""")
rep("(HOME / 'keep_v1.json').write_text(json.dumps(dict(keep=keep, utc=stamp, dirs=[str(DG)] + ([str(DF)] if spec else [])), indent=1))",
    "(HOME / 'keep_v1.json').write_text(json.dumps(dict(keep=keep, utc=stamp, dirs=[str(DG)] + ([str(DF)] if (DF / 'arms.json').exists() else [])), indent=1))")
rep("routes={f: r for f, r in sorted(routes.items()) if r}, route_table=ROUTES, dials=dials,",
    "routes={f: r for f, r in sorted(routes.items()) if r}, route_table=ROUTES, dials=dials, aggregate_rule='(d) one arm fagg: per family the largest-|t| band with "
    "t^2 >= 4 not overlapping a strict band, moved -sign(t) x 0.02 x min(1, |t|/6)', aggregate={f: dict(lo=v[0], hi=v[1], t=v[2]) for f, v in agg.items()},")
stamp = os.popen('date -u +%FT%TZ').read().strip()
assert s.startswith('\"\"\"')
s = ('\"\"\"%s (written %s, lead 17:0xZ / radlab): = %s with rule (d), ONE aggregate weak-band arm fagg (cert/RAD/make_agg_arm_v1.py: per family\n'
     'the largest-|t| routed band with t^2 >= 4 that rule (b) did not take and that overlaps none of its strict bands, moved against its gradient sign by\n'
     '0.02 x min(1, |t|/6), all in one arm).\n' % (dst.name, stamp, src.name)) + s[3:]
dst.write_text(s)
print('%s -> %s: rule (d) aggregate arm' % (src, dst))
