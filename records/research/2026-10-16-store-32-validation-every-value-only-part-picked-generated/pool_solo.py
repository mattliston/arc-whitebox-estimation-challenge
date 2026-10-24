"""[opus-solo] store-32 stats for any arm tag (res_d_<tag>_<r>.json vs res_ctl_<r>.json in opus_kinkoracle_v1) + optional seed-side numbers
from a GRADE_<name>.txt (grade_retune_v1 output) -> inverse-variance pooled xMSE and xnet. usage: pool_solo.py <tag>[=GRADE_name] ..."""
import json, math, re, statistics as st, sys
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
K = O/'lead/runs/opus_kinkoracle_v1'; S = O/'lead/runs/opus_solo_v1'
ld = lambda p: json.loads(p.read_text())
ctl = {int(p.stem.split('_')[-1]): ld(p) for p in K.glob('res_ctl_*.json')}
def store(tag):
    a = {int(p.stem.split('_')[-1]): ld(p) for p in K.glob(f'res_d_{tag}_*.json')}
    rs = sorted(set(a) & set(ctl))
    if len(rs) < 3: return None
    x = [100 * math.log(a[r]['U'] / ctl[r]['U']) for r in rs]; f = [100 * math.log(a[r]['flops'] / ctl[r]['flops']) for r in rs]
    t = [p + q for p, q in zip(x, f)]; se = lambda v: st.stdev(v) / math.sqrt(len(v))
    return dict(n=len(rs), x=st.mean(x), xse=se(x), bill=st.mean(f), t=st.mean(t), tse=se(t), lower=sum(v < 0 for v in x))
def seeds(name):
    p = S/f'GRADE_{name}.txt'
    if not p.exists(): return None
    g = {}
    for m in re.finditer(r'mean (\w+)\s+([+-][\d.]+)%\s+se ([\d.]+)\s+lower (\d+)/(\d+)', p.read_text()):
        g[m.group(1)] = (float(m.group(2)), float(m.group(3)), int(m.group(4)), int(m.group(5)))
    return g or None
def ivw(pairs):
    pairs = [(v, s) for v, s in pairs if v is not None and s]
    if not pairs: return None, None
    w = [1 / s ** 2 for _, s in pairs]
    return sum(wi * v for wi, (v, _) in zip(w, pairs)) / sum(w), 1 / math.sqrt(sum(w))
for a in sys.argv[1:]:
    tag, _, gname = a.partition('='); m = store(tag); g = seeds(gname or tag)
    sx = (g['xmse'][0], g['xmse'][1]) if g else (None, None); sn = (g['xnet'][0], g['xnet'][1]) if g else (None, None)
    px, pxs = ivw([sx, (m['x'], m['xse']) if m else (None, None)]); pn, pns = ivw([sn, (m['t'], m['tse']) if m else (None, None)])
    print(f"{tag:12s} seeds: " + (f"xMSE {sx[0]:+.3f}({sx[1]:.3f}) xnet {sn[0]:+.3f}({sn[1]:.3f}) bill {g['bill'][0]:+.3f} n{g['xmse'][3]}" if g else 'none') +
          " | store: " + (f"n{m['n']} xMSE {m['x']:+.3f}({m['xse']:.3f}) lower {m['lower']}/{m['n']} bill {m['bill']:+.3f} xnet {m['t']:+.3f}({m['tse']:.3f})" if m else 'none') +
          " | POOLED " + (f"xMSE {px:+.3f}({pxs:.3f}) xnet {pn:+.3f}({pns:.3f})" if px is not None else '-'))
