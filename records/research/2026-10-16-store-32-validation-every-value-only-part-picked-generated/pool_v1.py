"""[opus-storeval-u64] per-part pooled estimates, generated seeds (24: S8 7103248-55 vs ctl_s2tr + A80 7103080-095 vs ctl_s2tr_a80, from the
confirms in TEAM_LOG) + the 32 store ranks (hunter's harness: res_d_<tag>_<r>.json vs res_ctl_<r>.json, noise-free 2 x 32M truths, cold single
predict), inverse-variance on xMSE and on xnet. usage: pool_v1.py"""
import json, math, statistics as st
from pathlib import Path
K = Path(__file__).resolve().parents[1]/'opus_kinkoracle_v1'
ld = lambda p: json.loads(p.read_text())
ctl = {int(p.stem.split('_')[-1]): ld(p) for p in K.glob('res_ctl_*.json')}
def arm(tag): return {int(p.stem.split('_')[-1]): ld(p) for p in K.glob(f'res_d_{tag}_*.json')}
def store(a, b):
    rs = sorted(set(a) & set(b))
    if len(rs) < 3: return None
    x = [100 * math.log(a[r]['U'] / b[r]['U']) for r in rs]; f = [100 * math.log(a[r]['flops'] / b[r]['flops']) for r in rs]
    t = [p + q for p, q in zip(x, f)]; se = lambda v: st.stdev(v) / math.sqrt(len(v))
    return dict(n=len(rs), x=st.mean(x), xse=se(x), bill=st.mean(f), t=st.mean(t), tse=se(t))
# seed-side confirms (24 seeds; xMSE, its SE, bill, xnet, its SE) as posted in TEAM_LOG
SEEDS = {'dc35': (-0.234, .13, 0.065, -0.169, .13), 'l15rc30': (0.045, .117, -0.360, -0.315, .121), 'dcv2': (0.067, .122, -0.148, -0.082, .127),
         'stkdcv2rc30': (0.112, .14, -0.509, -0.397, .15), 'stk4': (None, None, -0.444, -0.828, .265), 'jord': (-0.236, .19, 0.0, -0.236, .19),
         'tankb10': (-0.451, .218, 0.007, -0.444, .218)}   # B vs S2tr 24 seeds (release tank, 17:08Z); JORD bill 0 so xMSE = xnet
def ivw(pairs):
    pairs = [(v, s) for v, s in pairs if v is not None and s]
    if not pairs: return None, None
    w = [1 / s ** 2 for _, s in pairs]
    return sum(wi * v for wi, (v, _) in zip(w, pairs)) / sum(w), 1 / math.sqrt(sum(w))
print(f"{'part':12s} {'seeds xMSE':>14s} {'store xMSE':>16s} {'POOLED xMSE':>14s} | {'seeds xnet':>14s} {'store xnet':>16s} {'POOLED xnet':>14s} | bill seeds/store")
for tag in ('dc35', 'l15rc30', 'dcv2', 'stkdcv2rc30', 'stk4', 'jord', 'tankb10', 'ukordb1'):
    s = SEEDS.get(tag); m = store(arm(tag), ctl)
    sx = f"{s[0]:+.3f}({s[1]:.3f})" if s and s[0] is not None else '      -       '
    st_x = f"{m['x']:+.3f}({m['xse']:.3f})n{m['n']}" if m else '       -        '
    px, pxs = ivw([(s[0], s[1]) if s else (None, None), (m['x'], m['xse']) if m else (None, None)])
    sn = f"{s[3]:+.3f}({s[4]:.3f})" if s else '      -       '
    st_n = f"{m['t']:+.3f}({m['tse']:.3f})n{m['n']}" if m else '       -        '
    pn, pns = ivw([(s[3], s[4]) if s else (None, None), (m['t'], m['tse']) if m else (None, None)])
    print(f"{tag:12s} {sx:>14s} {st_x:>16s} {('%+.3f(%.3f)' % (px, pxs)) if px is not None else '-':>14s} | {sn:>14s} {st_n:>16s} {('%+.3f(%.3f)' % (pn, pns)) if pn is not None else '-':>14s} | {s[2] if s else '-'} / {('%+.3f' % m['bill']) if m else '-'}")
