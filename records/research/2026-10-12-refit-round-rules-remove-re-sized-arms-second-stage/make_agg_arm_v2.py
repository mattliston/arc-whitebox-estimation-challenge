"""TAG opus-trainer RAD (10-12 17:0xZ; radlab 17:0xZ: the aggregate weak-band arm as radlab's FROZEN builder O/radlab/aggregate_spec_v1.py, sha
d0ab129b619a0fbc..., numpy only): writes the next version of a round's make file (one made by make_agg_arm_v1.py) whose rule (d) takes its ONE
aggregate arm 'fagg' from radlab's builder instead of my v1 rule. The builder runs on the round's own 64 gradients and the previous round's
keep file. Its rule: |t| >= 2, not strict, not overlapping a strict band of the family, not overlapping a previous-round dial band, the stronger of
overlapping weak bands. Each component is moved against its gradient sign by its jmix dial trial (-0.5% first order, cap 0.02) x |t| / sum |t|.
The make file applies that 'spec' as one arm file (start + all components).
If the builder file's sha256 differs from the frozen one, the round gets no aggregate arm (noted in RULES).
usage (from O/trainer): python -B cert/RAD/make_agg_arm_v2.py <make in> <make out (new file)>"""
import os
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
assert not dst.exists(), dst
s = src.read_text()
a = s.index("    agg = {}      # (d) ONE aggregate weak-band arm 'fagg' (lead 17:0xZ, radlab; cert/RAD/make_agg_arm_v1.py)\n")
b = s.index("    keep = read(DG / 'arms.json')['live'] + (read(DF / 'arms.json')['live'] if (DF / 'arms.json').exists() else [])\n")
new = """    agg, agg_note = {}, ''      # (d) ONE aggregate weak-band arm 'fagg' from radlab's frozen builder (cert/RAD/make_agg_arm_v2.py)
    import hashlib
    AGG_SCRIPT, AGG_SHA = Path('../radlab/aggregate_spec_v1.py'), 'd0ab129b619a0fbcf66d2f5c254ac1a481f2d15035ca98aa5604f08ac6369b0b'
    aggf, prevk = HOME / 'aggregate_spec_v1.json', Path('cert/RAD/S%d/keep_v1.json' % (int(HOME.name[1:]) - 1))
    if spec:
        agg_note = 'none: rule (c) arms present'
    elif not AGG_SCRIPT.exists() or hashlib.sha256(AGG_SCRIPT.read_bytes()).hexdigest() != AGG_SHA:
        agg_note = 'none: radlab/aggregate_spec_v1.py missing or not the frozen sha'
    else:
        if not aggf.exists():
            subprocess.run([PY, '-B', str(AGG_SCRIPT), 'run_RAD_%s_gen/results' % HOME.name, 'a_R%sg' % HOME.name, str(aggf)] + ([str(prevk)] if prevk.exists() else []), check=True)
        sp = read(aggf).get('spec') or ''
        comps = [x.split(':') for x in sp.split(',') if x]
        agg_note = 'radlab/aggregate_spec_v1.py (sha %s...): %s' % (AGG_SHA[:16], sp or 'no weak band: no arm')
        if comps:
            sys.path.insert(0, '.')
            from otr_common import load_params, load_pl, save_params
            A0, d0 = load_params(torch, cfg['params']); pl = {k: [float(x) for x in v] for k, v in load_pl(torch, cfg['params']).items()}
            for fam, lo, hi, sh in comps:
                lo, hi, sh = int(lo), int(hi), float(sh)
                pl[fam] = [x + (sh if lo <= l <= hi else 0.0) for l, x in enumerate(pl.get(fam, [0.0] * 16))]
                agg[fam] = (lo, hi, sh)
            DF.mkdir(parents=True)
            save_params(torch, DF / 'arm_fagg.pt', A0, d0, pl); (DF / 'arm_fagg_hu.json').write_text(json.dumps(read(cfg['hu'])))
            (DF / 'arms.json').write_text(json.dumps(dict(live=['fagg'], size=dict(fagg=1.0), base=cfg['params'], base_hu=cfg['hu'], spec=sp,
                                                          source=agg_note, aggregate_spec=str(aggf)), indent=1))
"""
s = s[:a] + new + s[b:]
old = "aggregate_rule='(d) one arm fagg: per family the largest-|t| band with t^2 >= 4 not overlapping a strict band, moved -sign(t) x 0.02 x min(1, |t|/6)', aggregate={f: dict(lo=v[0], hi=v[1], t=v[2]) for f, v in agg.items()},"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "aggregate_rule='(d) one arm fagg from radlab/aggregate_spec_v1.py (frozen sha d0ab129b...)', aggregate={f: dict(lo=v[0], hi=v[1], shift=v[2]) for f, v in agg.items()}, aggregate_note=agg_note,")
stamp = os.popen('date -u +%FT%TZ').read().strip()
assert s.startswith('\"\"\"')
s = ('\"\"\"%s (written %s, radlab 17:0xZ): = %s with rule (d) taken from the frozen radlab aggregate builder (cert/RAD/make_agg_arm_v2.py).\n' % (dst.name, stamp, src.name)) + s[3:]
dst.write_text(s)
print('%s -> %s: rule (d) from radlab/aggregate_spec_v1.py' % (src, dst))
