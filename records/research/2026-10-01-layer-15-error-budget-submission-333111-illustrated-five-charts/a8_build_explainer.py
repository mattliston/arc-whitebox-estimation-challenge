#!/usr/bin/env python3
"""TAG analyst B: build explainers/47_where_the_last_error_lives.html (one self-contained file: inline CSS, inline SVG charts, a
small inline script for hover read-outs). Every number in the page is read from the analyst's result files:
  a10_final_layer_oracle.json (last-layer-only replacement), a5_leaders.json (board), a7_native40_profile.json (banked estimator, 40 validation networks, per-layer error),
  a3_current_pooled.json (injected error per input, four networks), a1_e1_summary.json (norm conditioning, four networks),
  e4_r35.txt / e4_r100.txt (per-unit transport test), a6_summary.json (oracle levers), a4_cov_anatomy_rank0035.json, a2_parity.json.
usage: a8_build_explainer.py [out.html]"""
import sys, json, re, html
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/Users/matt/arc-white-box-estimation/explainers/47_where_the_last_error_lives.html')
J = lambda name: json.load(open(HERE / name))
board, nat, a3, e1, a6, a4, par = J('a5_leaders.json'), J('a7_native40_profile.json'), J('a3_current_pooled.json'), J('a1_e1_summary.json'), J('a6_summary.json'), J('a4_cov_anatomy_rank0035.json'), J('a2_parity.json')
NETS = ['35', '100', '37', '39']
a10 = J('a10_final_layer_oracle.json')['final']; f15 = lambda k: float(np.mean([a10[n][k] for n in NETS]))   # inputs exact at layer 15 only

# ---------------------------------------------------------------- numbers used in the prose
def sci(x, d=3):
    m, e = f'{x:.{d}e}'.split('e'); return f'{m}e{int(e)}'
def z(x, nd=3, sign=False):
    t = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    return f'{0:.{nd}f}' if float(t) == 0 else t
def pct(x, nd=0):
    t = f'{x:+.{nd}f}'
    return (f'{0:.{nd}f}' if float(t) == 0 else t.replace('-', '\u2212')) + '%'
from datetime import datetime
_now = datetime.fromisoformat(board['now']); read_at = f'{_now.day} {_now:%B %Y} at {_now:%H:%M} UTC'
T = {t['team']: t for t in board['teams']}
order = ['J2W', 'marius_binner', 'Luna', 'suliman_tadros', 'mliston']
label = {'J2W': 'J2W', 'marius_binner': 'marius_binner', 'Luna': 'Luna', 'suliman_tadros': 'suliman_tadros', 'mliston': 'us'}
us, first, third = T['mliston']['now'], T['J2W']['now'], T['Luna']['now']
early = lambda n: (16 * n['all_mse'] - n['final_mse']) / 15          # mean error of layers 0-14 from the all-layer and final-layer averages
er = {k: (early(T[k]['now']) / early(us), T[k]['now']['final_mse'] / us['final_mse']) for k in order}
gap_score = us['score'] / first['score'] - 1; gap_err = us['final_mse'] / first['final_mse'] - 1; gap_share = us['share'] / first['share'] - 1
need = lambda target: (1 - target / us['score']) * 100
mar = T['marius_binner']
rows3 = a3['rows']
deep = [r for r in rows3 if 10 <= r['layer'] <= 15]
rng = lambda k, rr=deep, s=1e9, d=1: f'{min(r[k] for r in rr) * s:.{d}f} to {max(r[k] for r in rr) * s:.{d}f}'
ret = [r['retention'] for r in rows3 if 7 <= r['layer'] <= 15]
lrn = [r['learned_removes'] for r in rows3 if 6 <= r['layer'] <= 15]
fresh_ratio = np.mean([r['fresh'] / (r['var'] + r['skew'] + r['kurt']) for r in deep])
ret_mean = float(np.mean(ret))
A6 = a6['final']; a6m = lambda k: float(np.mean([A6[str(r)][k] for r in a6['ranks']])) * 1e9
a6r = lambda k: (min(A6[str(r)][k] for r in a6['ranks']) * 1e9, max(A6[str(r)][k] for r in a6['ranks']) * 1e9)
e1m = lambda l, k: float(np.mean([e1[n]['rows'][l][k] for n in NETS]))
e1r = lambda l, k: (min(e1[n]['rows'][l][k] for n in NETS), max(e1[n]['rows'][l][k] for n in NETS))
def e4_rows(rank):
    out = []
    for line in open(HERE / f'e4_r{rank}.txt'):
        m = re.match(r'\s*(\d+)->\s*(\d+) \|', line)
        if not m: continue
        p = [x.strip() for x in line.split('|')]
        out.append(dict(a=int(m.group(1)), b=int(m.group(2)), rel=float(p[2]), r2=float(p[4]), skew_r2=float(p[5].split()[1])))
    return out
E4 = {35: e4_rows(35), 100: e4_rows(100)}
e4max = max(r['r2'] for k in E4 for r in E4[k]); e4min = min(r['r2'] for k in E4 for r in E4[k]); e4skew = max(r['skew_r2'] for k in E4 for r in E4[k])
e4rel = (min(r['rel'] for k in E4 for r in E4[k]), max(r['rel'] for k in E4 for r in E4[k]))
L13 = [x for x in a4['layers'] if x['layer'] == 13][0]; L6 = [x for x in a4['layers'] if x['layer'] == 6][0]; L10 = [x for x in a4['layers'] if x['layer'] == 10][0]
par_mean = np.mean([x['torch_final'] for x in par['rows']]) / np.mean([x['native_final'] for x in par['rows']]) - 1
par_corr = (min(min(x['corr_l12_15']) for x in par['rows']), max(max(x['corr_l12_15']) for x in par['rows']))
_gap = {x['rank']: x['ratio_pct'] for x in par['rows']}; gap_lo, gap_hi = min(-_gap[r] for r in (35, 37, 39)), max(-_gap[r] for r in (35, 37, 39)); gap_100 = _gap[100]
assert gap_lo > 0 and gap_100 > 0
mean40 = np.array(nat['mean']) * 1e9; slope = float(np.polyfit(np.arange(1, 16), mean40[1:], 1)[0]); steps = np.diff(mean40)
floor = [r['truth_floor'] * 1e9 for r in rows3]
mc_n = 2 ** 41 * 0.1 / (16 * 1024 * 1024 * 2); mc_mse = floor[15] / mc_n   # floor[15] (x1e9 / 1e9 samples) = mean output variance of a unit at layer 15

# ---------------------------------------------------------------- SVG helpers
def esc(s): return html.escape(str(s), quote=True)
def txt(x, y, s, cls='', anchor='start', size=11.5, extra=''):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" font-size="{size}" {extra}>{esc(s)}</text>'
def line(x1, y1, x2, y2, cls): return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>'
def path(pts, cls, extra=''): return f'<path d="M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts) + f'" class="{cls}" {extra}/>'
def hit(x, y, w, h, tip): return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" class="hit" tabindex="0" data-tip="{esc(tip)}"/>'
def bar_h(x, y, w, h, cls, r=4):   # horizontal bar from x, rounded right end
    w = max(w, r + 0.5)
    return f'<path d="M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z" class="{cls}"/>'
def col_seg(x, y, w, h, cls, top=False, r=4, extra=''):   # vertical segment; rounded top when it is the last of its stack
    if h <= 0.05: return ''
    if top and h > r:
        return f'<path d="M{x:.1f},{y + h:.1f} v-{h - r:.1f} a{r},{r} 0 0 1 {r},-{r} h{w - 2 * r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - r:.1f} z" class="{cls}" {extra}/>'
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" class="{cls}" {extra}/>'

# ---------------------------------------------------------------- chart 1: the board, three aligned panels
def chart_board():
    W, H = 920, 252; rows = [T[k]['now'] for k in order]; y0, rh, bh = 54, 34, 14
    panels = [('adjusted score', '×1e-9, lower is better', [r['score'] * 1e9 for r in rows], 2.5, [0, 1, 2], lambda v: f'{v:.3f}', 168, 196),
              ('final-layer error', '×1e-8', [r['final_mse'] * 1e8 for r in rows], 2.0, [0, 1, 2], lambda v: f'{v:.3f}', 430, 190),
              ('budget share', '% of the compute budget, minimum 10', [r['share'] * 100 for r in rows], 16.0, [0, 5, 10, 15], lambda v: f'{v:.1f}', 688, 176)]
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Top five teams: adjusted score equals final-layer error times budget share. We are fifth at 2.067e-9 with the largest error, 1.712e-8, and the second-smallest budget share, 12.1 percent.">']
    for i, k in enumerate(order):
        y = y0 + i * rh
        s.append(txt(150, y + bh - 2.5, f'{i + 1}  {label[k]}', 'tk' + (' b' if k == 'mliston' else ''), 'end', 12))
    for name, unit, vals, vmax, ticks, fmt, x0, pw in panels:
        sx = lambda v: x0 + v / vmax * pw
        s.append(txt(x0, 18, name, 'tk b', 'start', 12.5)); s.append(txt(x0, 33, unit, 't2', 'start', 10.5))
        for t in ticks:
            s.append(line(sx(t), y0 - 8, sx(t), y0 + 5 * rh - 12, 'grid'))
            s.append(txt(sx(t), y0 + 5 * rh + 4, f'{t:g}', 't2', 'middle', 10.5))
        if name == 'budget share':
            s.append(line(sx(10), y0 - 12, sx(10), y0 + 5 * rh - 12, 'ref')); s.append(txt(sx(10) + 4, y0 + 5 * rh + 18, 'minimum 10%', 't2', 'middle', 10.5))
        s.append(line(x0, y0 - 8, x0, y0 + 5 * rh - 12, 'axis'))
        for i, (k, v) in enumerate(zip(order, vals)):
            y = y0 + i * rh
            s.append(bar_h(x0, y, sx(v) - x0, bh, 'm-us' if k == 'mliston' else 'm-other'))
            s.append(txt(sx(v) + 6, y + bh - 2.5, fmt(v), 'tk' + (' b' if k == 'mliston' else ''), 'start', 11.5))
    s.append(txt(399, y0 + 2 * rh + 12, '=', 'tk', 'middle', 22)); s.append(txt(660, y0 + 2 * rh + 12, '×', 'tk', 'middle', 22))
    s.append('</svg>'); return '\n'.join(s)

# ---------------------------------------------------------------- chart 3: per-layer error, 40 networks
def chart_layers():
    W, H = 920, 392; x0, x1, y0, y1 = 62, 838, 24, 326; ymax = 24.0
    sx = lambda l: x0 + l / 15 * (x1 - x0); sy = lambda v: y1 - v / ymax * (y1 - y0)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Error of the board estimator at each layer on 40 validation networks. The mean rises from 0.7e-9 at layer 0, which is the noise of the truth, to 17.2e-9 at layer 15, about 1.2e-9 per layer.">']
    for t in range(0, 25, 4):
        s.append(line(x0, sy(t), x1, sy(t), 'grid' if t else 'axis')); s.append(txt(x0 - 8, sy(t) + 4, f'{t}', 't2', 'end', 11))
    for l in range(16): s.append(txt(sx(l), y1 + 18, f'{l}', 't2', 'middle', 11))
    s.append(txt((x0 + x1) / 2, y1 + 40, 'layer (0 = first, 15 = last; only layer 15 is scored)', 't2', 'middle', 11.5))
    s.append(txt(16, (y0 + y1) / 2, 'mean-squared error of the unit means (×1e-9)', 't2', 'middle', 11.5, f'transform="rotate(-90 16 {(y0 + y1) / 2:.1f})"'))
    for rank, prof in nat['per_network'].items():
        s.append(path([(sx(l), sy(v * 1e9)) for l, v in enumerate(prof)], 'l-thin'))
    s.append(path([(sx(l), sy(v)) for l, v in enumerate(floor)], 'l-floor'))
    s.append(path([(sx(l), sy(v)) for l, v in enumerate(mean40)], 'l-mean'))
    s.append(f'<circle cx="{sx(15):.1f}" cy="{sy(mean40[15]):.1f}" r="5" class="dot-mean"/>')
    s.append(txt(sx(15) + 10, sy(mean40[15]) + 4, f'mean {mean40[15]:.1f}', 'tk b', 'start', 12))
    s.append(txt(sx(15) + 10, sy(floor[15]) - 5, f'floor {floor[15]:.2f}', 'tk', 'start', 11.5))
    s.append(txt(sx(15) + 10, sy(nat['hi'][15] * 1e9) + 4, f'highest {nat["hi"][15] * 1e9:.1f}', 't2', 'start', 11)); s.append(txt(sx(15) + 10, sy(nat['lo'][15] * 1e9) + 4, f'lowest {nat["lo"][15] * 1e9:.1f}', 't2', 'start', 11))
    s.append(f'<line x1="0" y1="{y0}" x2="0" y2="{y1}" class="cross" id="cross3" visibility="hidden"/>')
    for l in range(16):
        tip = f'layer {l}\nmean of 40 networks: {mean40[l]:.2f}e-9\nlowest / highest network: {nat["lo"][l] * 1e9:.2f} / {nat["hi"][l] * 1e9:.2f}e-9\nnoise floor of the truth: {floor[l]:.2f}e-9'
        s.append(f'<rect x="{sx(l) - 25.8:.1f}" y="{y0}" width="51.6" height="{y1 - y0}" class="hit" tabindex="0" data-cross="cross3" data-x="{sx(l):.1f}" data-tip="{esc(tip)}"/>')
    s.append('</svg>'); return '\n'.join(s)

# ---------------------------------------------------------------- chart 4: injected error per layer, stacked
def chart_inject():
    W, H = 920, 372; x0, x1, y0, y1 = 62, 812, 24, 306; ymax = 5.0; bw = 24
    sx = lambda l: x0 + (l - 0.5) / 15 * (x1 - x0); sy = lambda v: y1 - v / ymax * (y1 - y0)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="New error added at each layer by wrong variance, skew and kurtosis inputs, mean of four networks. At layer 15: variance 2.0, skew 1.2, kurtosis 1.3, combined 3.8, in units of 1e-9.">']
    for t in range(0, 6):
        s.append(line(x0, sy(t), x1, sy(t), 'grid' if t else 'axis')); s.append(txt(x0 - 8, sy(t) + 4, f'{t}', 't2', 'end', 11))
    s.append(txt((x0 + x1) / 2, y1 + 40, 'layer', 't2', 'middle', 11.5))
    s.append(txt(16, (y0 + y1) / 2, 'new mean-squared error added at this layer (×1e-9)', 't2', 'middle', 11.5, f'transform="rotate(-90 16 {(y0 + y1) / 2:.1f})"'))
    for r in rows3[1:]:
        l = r['layer']; x = sx(l) - bw / 2; s.append(txt(sx(l), y1 + 18, f'{l}', 't2', 'middle', 11))
        base = 0.0; parts = [('var', 'm-s1'), ('skew', 'm-s2'), ('kurt', 'm-s3')]
        for i, (k, cls) in enumerate(parts):
            v = max(r[k] * 1e9, 0.0); h = v / ymax * (y1 - y0); gap = 2 if base > 0 and h > 2.5 else 0
            s.append(col_seg(x, sy(base + v) , bw, h - gap, cls + (' pale' if r[k + '_at_floor'] else ''), top=(i == 2)))
            base += v
        if not (r['var_at_floor'] or r['skew_at_floor']):
            s.append(line(sx(l) - 17, sy(r['fresh'] * 1e9), sx(l) + 17, sy(r['fresh'] * 1e9), 'mk-comb'))
        fl = lambda k: ' (at the noise floor)' if r[k + '_at_floor'] else ''
        tip = (f'layer {l}\nvariance: {max(r["var"], 0) * 1e9:.2f}e-9{fl("var")}\nskew: {r["skew"] * 1e9:.2f}e-9{fl("skew")}\nkurtosis: {r["kurt"] * 1e9:.2f}e-9{fl("kurt")}\n'
               f'combined: {r["fresh"] * 1e9:.2f}e-9\ncarried in from layer {l - 1}: {r["carried"] * 1e9:.2f}e-9')
        s.append(hit(sx(l) - 27, y0, 54, y1 - y0, tip))
    r15 = rows3[15]; s.append(txt(sx(15) + 24, sy((r15['var'] + r15['skew'] + r15['kurt']) * 1e9) + 8, f'sum {(r15["var"] + r15["skew"] + r15["kurt"]) * 1e9:.1f}', 'tk', 'start', 11.5))
    s.append(txt(sx(15) + 24, sy(r15['fresh'] * 1e9) + 4, f'combined {r15["fresh"] * 1e9:.1f}', 'tk b', 'start', 11.5))
    s.append('</svg>'); return '\n'.join(s)

# ---------------------------------------------------------------- chart 5: norm conditioning, three panels
def chart_gain():
    W, H = 920, 318; s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Skew and kurtosis of unit inputs by layer, over all inputs and inside groups of inputs with equal previous-layer norm, mean of four networks. Grouping removes most of the skew spread and brings the average kurtosis to zero (part of that is the fixed-norm constraint), but not the unit-to-unit spread of the kurtosis.">']
    panels = [('Skew: spread across units', 'g1_spread', 'g1_within_spread', 0.12, [0, 0.04, 0.08, 0.12], 2),
              ('Excess kurtosis: mean over units', 'g2_mean', 'g2_within_mean', 0.07, [0, 0.02, 0.04, 0.06], 2),
              ('Excess kurtosis: spread across units', 'g2_spread', 'g2_within_spread', 0.02, [0, 0.005, 0.010, 0.015, 0.020], 3)]
    for i, (title, ka, kw, ymax, ticks, nd) in enumerate(panels):
        x0 = 54 + i * 306; x1 = x0 + 196; y0, y1 = 46, 250; ymin = -0.012 * (ymax / 0.07) if 'mean' in ka else 0.0
        sx = lambda l: x0 + (l - 1) / 14 * (x1 - x0); sy = lambda v: y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)
        s.append(txt(x0 - 30, 22, title, 'tk b', 'start', 12))
        for t in ticks:
            s.append(line(x0, sy(t), x1, sy(t), 'grid' if t else 'axis')); s.append(txt(x0 - 7, sy(t) + 4, f'{t:.{nd}f}', 't2', 'end', 10.5))
        for l in (1, 5, 10, 15): s.append(txt(sx(l), y1 + 17, f'{l}', 't2', 'middle', 10.5))
        s.append(txt((x0 + x1) / 2, y1 + 36, 'layer', 't2', 'middle', 11))
        a = [e1m(l, ka) for l in range(1, 16)]; w = [e1m(l, kw) for l in range(1, 16)]
        s.append(path([(sx(l), sy(v)) for l, v in zip(range(1, 16), a)], 'l-s1')); s.append(path([(sx(l), sy(v)) for l, v in zip(range(1, 16), w)], 'l-s2'))
        s.append(f'<circle cx="{sx(15):.1f}" cy="{sy(a[-1]):.1f}" r="4.5" class="dot-s1"/><circle cx="{sx(15):.1f}" cy="{sy(w[-1]):.1f}" r="4.5" class="dot-s2"/>')
        close = abs(sy(a[-1]) - sy(w[-1])) < 15
        s.append(txt(sx(15) + 9, sy(a[-1]) + (-3 if close else 4), z(a[-1]), 'tk', 'start', 11.5)); s.append(txt(sx(15) + 9, sy(w[-1]) + (13 if close else 4), z(w[-1]), 'tk', 'start', 11.5))
        s.append(f'<line x1="0" y1="{y0}" x2="0" y2="{y1}" class="cross" id="cross5{i}" visibility="hidden"/>')
        for l in range(1, 16):
            tip = f'layer {l}\nall inputs: {a[l - 1]:.4f}\ninside groups of equal norm: {w[l - 1]:.4f}'
            s.append(f'<rect x="{sx(l) - 7:.1f}" y="{y0}" width="14" height="{y1 - y0}" class="hit" tabindex="0" data-cross="cross5{i}" data-x="{sx(l):.1f}" data-tip="{esc(tip)}"/>')
    s.append('</svg>'); return '\n'.join(s)

# ---------------------------------------------------------------- diagram: one layer step
def diagram():
    W, H = 920, 356; s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="One layer step: the carried mean, covariance, third-order column bank and fourth-order number are multiplied by the layer weights to give each unit\'s input mean, spread, skew and kurtosis; a formula turns these into the mean after the ReLU; a learned correction is added; the result is carried to the next layer.">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahead"/></marker></defs>']
    ys = [36, 104, 172, 240]; bw1, bh = 236, 54
    left = [('mean of each unit', '1,024 numbers', 'bx'), ('covariance of every pair of units', '1,024 × 1,024 numbers', 'bx'),
            ('third order: a bank of columns', 'new columns added at every layer', 'bx'), ('fourth order', '1 number per layer', 'bx')]
    mid = [('input mean  μ', 'per unit', 'bx'), ('input spread  σ', 'per unit, read from the covariance', 'bx s1'), ('input skew', 'per unit, read from the bank', 'bx s2'),
           ('input kurtosis', 'the same value for all units', 'bx s3')]
    s.append(txt(14, 20, 'CARRIED FROM LAYER n − 1', 't2 cap', 'start', 10.5)); s.append(txt(346, 20, 'INPUTS OF EACH UNIT AT LAYER n', 't2 cap', 'start', 10.5)); s.append(txt(676, 20, 'OUTPUT OF LAYER n', 't2 cap', 'start', 10.5))
    for (a, b, c), y in zip(left, ys):
        s.append(f'<rect x="14" y="{y}" width="{bw1}" height="{bh}" rx="5" class="{c}"/>'); s.append(txt(26, y + 22, a, 'tk b', 'start', 12)); s.append(txt(26, y + 40, b, 't2', 'start', 10.8))
    for (a, b, c), y in zip(mid, ys):
        s.append(f'<rect x="346" y="{y}" width="232" height="{bh}" rx="5" class="{c}"/>'); s.append(txt(358, y + 22, a, 'tk b', 'start', 12)); s.append(txt(358, y + 40, b, 't2', 'start', 10.8))
        s.append(f'<line x1="{14 + bw1 + 3}" y1="{y + bh / 2}" x2="343" y2="{y + bh / 2}" class="flow" marker-end="url(#ah)"/>'); s.append(txt((14 + bw1 + 346) / 2, y + bh / 2 - 7, '× weights', 't2', 'middle', 10.5))
    fx, fw = 676, 230
    s.append(f'<rect x="{fx}" y="70" width="{fw}" height="{bh}" rx="5" class="bx"/>'); s.append(txt(fx + 12, 92, 'formula for the mean', 'tk b', 'start', 12)); s.append(txt(fx + 12, 110, 'of a ReLU unit (section 4)', 't2', 'start', 10.8))
    for y in ys: s.append(f'<path d="M578,{y + bh / 2} C628,{y + bh / 2} 624,97 {fx - 3},97" class="flow" marker-end="url(#ah)" fill="none"/>')
    s.append(f'<rect x="{fx}" y="156" width="{fw}" height="{bh}" rx="5" class="bx"/>'); s.append(txt(fx + 12, 178, '+ learned correction', 'tk b', 'start', 12)); s.append(txt(fx + 12, 196, 'network with 45,000 parameters', 't2', 'start', 10.8))
    s.append(f'<line x1="{fx + fw / 2}" y1="{70 + bh + 2}" x2="{fx + fw / 2}" y2="153" class="flow" marker-end="url(#ah)"/>')
    s.append(f'<rect x="{fx}" y="242" width="{fw}" height="{bh}" rx="5" class="bx strong"/>'); s.append(txt(fx + 12, 264, 'mean of each unit after layer n', 'tk b', 'start', 12)); s.append(txt(fx + 12, 282, 'this is what is scored at layer 15', 't2', 'start', 10.8))
    s.append(f'<line x1="{fx + fw / 2}" y1="{156 + bh + 2}" x2="{fx + fw / 2}" y2="239" class="flow" marker-end="url(#ah)"/>')
    s.append(f'<path d="M{fx + fw / 2},{242 + bh + 2} V330 H132 V{240 + bh + 4}" class="flow" marker-end="url(#ah)" fill="none"/>')
    s.append(txt(400, 324, 'carried to layer n + 1, together with the updated covariance, bank and fourth-order number', 't2', 'middle', 10.8))
    s.append('</svg>'); return '\n'.join(s)

# ---------------------------------------------------------------- tables
def table(head, body, cls=''):
    h = ''.join(f'<th class="{"num" if i else ""}">{c}</th>' for i, c in enumerate(head))
    b = ''.join('<tr>' + ''.join(f'<td class="{"num" if i else ""}">{c}</td>' for i, c in enumerate(r)) + '</tr>' for r in body)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'
pc = lambda x: '' if x is None else f'{x:+.1f}%'
board_tab = table(['team (board entry)', 'adjusted score', 'final-layer error', 'all-layer error', 'error averaged over layers 0 to 14', 'budget share', 'run time per network, mean / longest', 'score, change in 72 h', 'error, change in 72 h'],
    [[f'{"<b>us</b>" if k == "mliston" else esc(k)} <span class="dim">#{T[k]["now"]["id"]}</span>', sci(T[k]['now']['score']), sci(T[k]['now']['final_mse']), sci(T[k]['now']['all_mse'], 2), sci(early(T[k]['now']), 2), f'{T[k]["now"]["share"] * 100:.1f}%',
      f'{T[k]["now"]["wall_mean"]:.0f} s / {T[k]["now"]["wall_max"]:.0f} s', pc(T[k]['change_pct']['score']), pc(T[k]['change_pct']['final_mse'])] for k in order])
layer_tab = table(['layer'] + [str(l) for l in range(16)],
    [['mean of 40 networks'] + [f'{v:.2f}' for v in mean40], ['lowest network'] + [f'{v * 1e9:.2f}' for v in nat['lo']], ['highest network'] + [f'{v * 1e9:.2f}' for v in nat['hi']],
     ['noise floor of the truth'] + [f'{v:.2f}' for v in floor]], 'small')
fl = lambda r, k: ' *' if r[k + '_at_floor'] else ''
inj_tab = table(['layer', 'carried in', 'variance', 'skew', 'kurtosis', 'three combined', 'research copy, before learned correction', 'research copy, after learned correction', 'board version, same four networks', 'learned correction removes', 'share of previous layer\'s error carried in'],
    [[r['layer'], f'{r["carried"] * 1e9:.2f}', f'{max(r["var"], 0) * 1e9:.2f}{fl(r, "var")}', f'{r["skew"] * 1e9:.2f}{fl(r, "skew")}', f'{r["kurt"] * 1e9:.2f}{fl(r, "kurt")}', f'{r["fresh"] * 1e9:.2f}' + (' *' if r['var_at_floor'] or r['skew_at_floor'] else ''),
      f'{r["host"] * 1e9:.2f}', f'{r["card"] * 1e9:.2f}', f'{r["native"] * 1e9:.2f}', f'{r["learned_removes"] * 100:.0f}%' + (' *' if r['learned_at_floor'] else ''), '' if r['retention'] is None else f'{r["retention"] * 100:.0f}%'] for r in rows3[1:]], 'small')
rel_tab = table(['layer', 'variance: relative error per unit', 'skew: error as a share of the true unit-to-unit spread', 'kurtosis: value the estimator uses', 'kurtosis: true average', 'kurtosis: true unit-to-unit spread'],
    [[r['layer'], f'{r["rel_v_err"] * 100:.3f}%', f'{r["g1_err_rel"] * 100:.1f}%', f'{r["g2_card"]:.4f}', f'{r["g2_mean"]:.4f}', f'{r["g2_spread"]:.4f}'] for r in rows3[1:] if r['layer'] in (1, 3, 5, 7, 9, 11, 13, 15)], 'small')
names6 = dict(base='none (the estimator as it is)', v='variance', s='skew', k='kurtosis', vk='variance and kurtosis', vs='variance and skew', sk='skew and kurtosis', vsk='all three')
b_on = a6m('base')
ora_tab = table(['inputs replaced by their true values at every layer', 'learned correction on: final-layer error', 'change', 'learned correction off: final-layer error', 'change'],
    [[names6[k], f'{a6m(k):.1f}e-9', '' if k == 'base' else pct((a6m(k) / b_on - 1) * 100), f'{a6m(k + "_learned_off"):.1f}e-9', pct((a6m(k + "_learned_off") / b_on - 1) * 100)] for k in ('base', 'v', 's', 'k', 'vs', 'vk', 'sk', 'vsk')])
e1_tab = table(['layer', 'skew spread, all inputs', 'skew spread, inside groups', 'average kurtosis, all inputs', 'average kurtosis, inside groups', 'kurtosis spread, all inputs', 'kurtosis spread, inside groups'],
    [[l, f'{e1m(l, "g1_spread"):.4f}', f'{e1m(l, "g1_within_spread"):.4f}', f'{e1m(l, "g2_mean"):+.4f}', f'{e1m(l, "g2_within_mean"):+.4f}', f'{e1m(l, "g2_spread"):.4f}', f'{e1m(l, "g2_within_spread"):.4f}'] for l in range(1, 16)], 'small')
e4_tab = table(['from layer → to layer', 'kurtosis, network 35: R²', 'kurtosis, network 100: R²', 'skew, network 35: R²', 'skew, network 100: R²', 'two independent truths of the target agree (correlation, network 35)'],
    [[f'{a["a"]} → {a["b"]}', f'{a["r2"]:.2f}', f'{b["r2"]:.2f}', f'{a["skew_r2"]:.3f}', f'{b["skew_r2"]:.3f}', f'{a["rel"]:.3f}'] for a, b in zip(E4[35], E4[100])], 'small')
best_err = mar['now']['final_mse']
others = sorted(t['now']['score'] for t in board['teams'] if t['team'] != 'mliston')
def place(score):
    n = 1 + sum(1 for x in others if x < score); return {1: '1st', 2: '2nd', 3: '3rd'}.get(n, f'{n}th')
sc_rows = [['today (our board entry)', sci(us['final_mse']), f'{us["share"] * 100:.1f}%', sci(us['score']), place(us['score'])],
           ['our error, budget share at the 10% minimum', sci(us['final_mse']), '10.0%', sci(us['final_mse'] * 0.1), place(us['final_mse'] * 0.1)],
           ['the current target: a bill cut worth 3.5%, error unchanged', sci(us['final_mse']), f'{us["share"] * 100 * 0.965:.1f}%', sci(us['score'] * 0.965), place(us['score'] * 0.965)],
           ['the best error on the board (marius_binner) at our budget share', sci(best_err), f'{us["share"] * 100:.1f}%', sci(best_err * us['share']), place(best_err * us['share'])],
           ['the best error on the board at the 10% minimum', sci(best_err), '10.0%', sci(best_err * 0.1), place(best_err * 0.1)],
           ['1e-9: an error 42% below ours, at the 10% minimum', '1.000e-8', '10.0%', '1.000e-9', '1st']]
sc_tab = table(['case', 'final-layer error', 'budget share', 'adjusted score', 'place on today\'s board'], sc_rows)

# ---------------------------------------------------------------- page
CSS = '''
:root{--bg:#fbfaf8;--panel:#f3f1ec;--ink:#1d212a;--ink2:#575d6b;--line:#e3e0d9;--axis:#c3c2b7;--muted:#898781;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s1t:#1c5cab}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171c;--panel:#1d2026;--ink:#e8e7e2;--ink2:#9ba0ac;--line:#2b2e36;--axis:#383835;--muted:#898781;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s1t:#7fb3f0}}
:root[data-theme="dark"]{--bg:#15171c;--panel:#1d2026;--ink:#e8e7e2;--ink2:#9ba0ac;--line:#2b2e36;--axis:#383835;--muted:#898781;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s1t:#7fb3f0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Source Serif 4",Georgia,"Times New Roman",serif;font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:720px;margin:0 auto;padding:0 20px}.wide{max-width:980px;margin:0 auto;padding:0 20px}
h1,h2{font-family:"STIX Two Text",Georgia,"Times New Roman",serif;font-weight:600;line-height:1.2;margin:0;text-wrap:balance}
h1{font-size:40px;letter-spacing:-.01em}h2{font-size:26px;margin-top:6px}
p{margin:14px 0}.lead{font-size:18.5px}
.sans,.eyebrow,.chip,figcaption,th,.legend,details summary,svg text,#tt{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.eyebrow{font-size:11.5px;font-weight:600;letter-spacing:.13em;text-transform:uppercase;color:var(--ink2)}
header{padding:56px 0 6px}.dek{font-size:19.5px;font-style:italic;color:var(--ink2);margin-top:14px;text-wrap:balance}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0 6px}.chip{font-size:11.5px;padding:5px 10px;border:1px solid var(--line);border-radius:3px;color:var(--ink2)}.chip b{color:var(--ink);font-weight:600}
section{padding:30px 0 6px}.sec-head{border-top:1px solid var(--line);padding-top:18px}.sec-head .eyebrow{display:block;margin-bottom:6px}
.eq{font-family:"STIX Two Text",Georgia,serif;font-size:18px;line-height:1.9;padding:14px 20px;margin:16px 0;background:var(--panel);border-radius:4px;overflow-x:auto;white-space:nowrap}
var{font-style:italic}
figure{margin:24px 0}figure svg{display:block;width:100%;height:auto}
.fig-frame{border:1px solid var(--line);border-radius:6px;padding:16px 12px 8px;overflow-x:auto;background:var(--bg)}.fig-frame svg{min-width:640px}
figcaption{font-size:12.5px;line-height:1.55;color:var(--ink2);margin-top:10px;padding:0 4px}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;margin:0 4px 10px;font-size:12px;color:var(--ink2)}
.legend i{display:inline-block;vertical-align:middle;margin-right:7px}
.k-rect{width:12px;height:12px;border-radius:2px}.k-line{width:22px;height:0;border-top:2.5px solid}.k-thin{width:22px;height:0;border-top:1px solid;opacity:.55}
svg text{fill:var(--ink)}svg .t2{fill:var(--ink2)}svg .b{font-weight:600}svg .cap{letter-spacing:.1em}
svg .grid{stroke:var(--line);stroke-width:1}svg .axis{stroke:var(--axis);stroke-width:1}svg .ref{stroke:var(--ink2);stroke-width:1}
svg .m-us{fill:var(--s1)}svg .m-other{fill:var(--muted)}svg .m-s1{fill:var(--s1)}svg .m-s2{fill:var(--s2)}svg .m-s3{fill:var(--s3)}svg .pale{opacity:.33}
svg .mk-comb{stroke:var(--ink);stroke-width:2}
svg .l-thin{fill:none;stroke:var(--s1);stroke-width:1;opacity:.22}svg .l-mean,svg .l-s1{fill:none;stroke:var(--s1);stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
svg .l-floor,svg .l-s2{fill:none;stroke:var(--s2);stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
svg .dot-mean,svg .dot-s1{fill:var(--s1);stroke:var(--bg);stroke-width:2}svg .dot-s2{fill:var(--s2);stroke:var(--bg);stroke-width:2}
svg .cross{stroke:var(--ink2);stroke-width:1}svg .hit{fill:transparent;outline:none}svg .hit:hover,svg .hit:focus{fill:var(--ink);fill-opacity:.05}
svg .bx{fill:var(--bg);stroke:var(--ink2);stroke-width:1.2}svg .bx.s1{stroke:var(--s1);stroke-width:2.2}svg .bx.s2{stroke:var(--s2);stroke-width:2.2}svg .bx.s3{stroke:var(--s3);stroke-width:2.2}svg .bx.strong{stroke:var(--ink);stroke-width:2}
svg .flow{stroke:var(--ink2);stroke-width:1.3;fill:none}svg .ahead{fill:var(--ink2)}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14.5px}
th{font-size:11px;letter-spacing:.04em;color:var(--ink2);text-align:left;font-weight:600;border-bottom:1px solid var(--ink2);padding:6px 10px 6px 0;vertical-align:bottom}
td{border-bottom:1px solid var(--line);padding:7px 10px 7px 0;vertical-align:top}
td.num,th.num{text-align:right;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;font-variant-numeric:tabular-nums}td.num{white-space:nowrap}th.num{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:11px}
table.small td{font-size:12.5px;padding:5px 8px 5px 0}table.small td.num{font-size:12px}
.tscroll{overflow-x:auto}.dim{color:var(--ink2);font-size:.85em}
details{margin:12px 0 4px;border:1px solid var(--line);border-radius:4px;padding:8px 14px}details summary{font-size:12px;cursor:pointer;color:var(--ink2)}
.note{background:var(--panel);border-radius:4px;padding:12px 18px;margin:18px 0;font-size:15.5px}
#tt{position:fixed;z-index:10;display:none;max-width:320px;background:var(--ink);color:var(--bg);font-size:12px;line-height:1.5;padding:8px 11px;border-radius:4px;pointer-events:none;white-space:pre-line}
footer{border-top:1px solid var(--line);margin-top:44px;padding:22px 0 64px;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:12px;line-height:1.75;color:var(--ink2)}
footer code{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;overflow-wrap:anywhere}.nw{white-space:nowrap}
a{color:var(--s1t)}
@media (max-width:640px){h1{font-size:31px}body{font-size:16px}}
'''
JS = '''
(function(){var tt=document.getElementById('tt');
function show(e,el){tt.textContent=el.getAttribute('data-tip');tt.style.display='block';var r=el.getBoundingClientRect();
 var x=(e&&e.clientX!==undefined&&e.clientX)?e.clientX+14:r.left+r.width/2+10,y=(e&&e.clientY!==undefined&&e.clientY)?e.clientY+14:r.top+20;
 if(x+tt.offsetWidth>window.innerWidth-8)x=Math.max(8,x-tt.offsetWidth-28);if(y+tt.offsetHeight>window.innerHeight-8)y=Math.max(8,y-tt.offsetHeight-28);
 tt.style.left=x+'px';tt.style.top=y+'px';var c=el.getAttribute('data-cross');
 if(c){var l=document.getElementById(c);l.setAttribute('x1',el.getAttribute('data-x'));l.setAttribute('x2',el.getAttribute('data-x'));l.setAttribute('visibility','visible');}}
function hide(el){tt.style.display='none';var c=el.getAttribute('data-cross');if(c)document.getElementById(c).setAttribute('visibility','hidden');}
document.querySelectorAll('.hit').forEach(function(el){el.addEventListener('pointermove',function(e){show(e,el)});el.addEventListener('pointerleave',function(){hide(el)});
 el.addEventListener('focus',function(){show(null,el)});el.addEventListener('blur',function(){hide(el)});});})();
'''
def fig(svg, legend, caption, tab=None, tabname='Table of the plotted values'):
    lg = f'<div class="legend">{legend}</div>' if legend else ''
    t = f'<details><summary>{tabname}</summary><div class="tscroll">{tab}</div></details>' if tab else ''
    return f'<figure class="wide"><div class="fig-frame">{lg}{svg}</div><figcaption>{caption}</figcaption>{t}</figure>'
def sec(n, eyebrow, title, body): return f'<section><div class="wrap sec-head"><span class="eyebrow">{n:02d} · {eyebrow}</span><h2>{title}</h2></div>{body}</section>'
P = lambda s: f'<div class="wrap"><p>{s}</p></div>'
r15, r13, r10 = rows3[15], rows3[13], rows3[10]
vlo, vhi = min(r['rel_v_err'] for r in deep) * 100, max(r['rel_v_err'] for r in deep) * 100
slo, shi = min(r['g1_err_rel'] for r in deep) * 100, max(r['g1_err_rel'] for r in deep) * 100
klow = [(1 - r['g2_card'] / r['g2_mean']) * 100 for r in rows3 if 8 <= r['layer'] <= 15]
koff = [r['kurt_offset_part'] / r['kurt_raw'] * 100 for r in rows3 if 8 <= r['layer'] <= 15]
ft = [r['formula_truth'] * 1e9 for r in rows3[1:]]
all_off = a6r('vsk_learned_off')
def pr(k):
    ch = [(A6[str(r)][k] / A6[str(r)]['base'] - 1) * 100 for r in a6['ranks']]
    lo, hi = -max(ch), -min(ch)     # reductions, smallest to largest
    return f'{max(lo, 0):.0f}% to {hi:.0f}%' if lo >= -0.5 else f'from {hi:.0f}% lower to {-lo:.0f}% higher'

S1 = (P(f'<span class="lead">We are fifth with an adjusted score of {sci(us["score"])}; first place is {sci(first["score"])} (J2W) and third place is {sci(third["score"])} (Luna).</span> '
        f'The adjusted score is the final-layer error multiplied by the share of the compute budget used. We have the largest error of the top five ({sci(us["final_mse"])}) and the second-smallest budget share ({us["share"] * 100:.1f}%). '
        f'marius_binner has the smallest error by a wide margin ({sci(mar["now"]["final_mse"])}) and the largest budget share ({mar["now"]["share"] * 100:.1f}%). J2W is first because its budget share ({first["share"] * 100:.1f}%) is closest to the 10% minimum.')
      + fig(chart_board(), '<span><i class="k-rect" style="background:var(--s1)"></i>us</span><span><i class="k-rect" style="background:var(--muted)"></i>the four teams ahead of us</span>',
            f'Board entries read from the public submission pages on {read_at} (the best graded entry of each team). Each score is the average over the 50 public board networks of error × budget share, which is why the product of the two averages shown here differs from the score in the last digit. Run times are over all 100 networks of the run.',
            f'<div class="tscroll">{board_tab}</div>', 'Table: the five entries and their change in 72 hours')
      + P(f'The board also publishes the error averaged over all 16 layers, which shows where each team is more accurate than we are. Averaged over layers 0 to 14, J2W\'s error is {er["J2W"][0] * 100:.0f}% of ours and its final-layer error is {er["J2W"][1] * 100:.0f}% of ours: its whole accuracy advantage is at the last layer. '
          f'marius_binner is at {er["marius_binner"][0] * 100:.0f}% and {er["marius_binner"][1] * 100:.0f}%, Luna at {er["Luna"][0] * 100:.0f}% and {er["Luna"][1] * 100:.0f}%, suliman_tadros at {er["suliman_tadros"][0] * 100:.0f}% and {er["suliman_tadros"][1] * 100:.0f}%: more accurate at every layer.')
      + P(f'Our gap to first place is {gap_score * 100:.0f}%: {gap_err * 100:.0f}% comes from the error and {gap_share * 100:.0f}% from the budget share. '
          f'Reaching 2.0e-9 needs our score to fall by {need(2e-9):.1f}%, third place needs {need(third["score"]):.1f}%, and 1.0e-9 needs {need(1e-9):.1f}%.')
      + P(f'In the 72 hours before the reading, marius_binner lowered its error by {-mar["change_pct"]["final_mse"]:.1f}% and its score by {-mar["change_pct"]["score"]:.1f}% over {mar["graded_in_window"]} graded entries, with the budget share down {-mar["change_pct"]["share"]:.1f}%. '
          f'J2W lowered its score by {-T["J2W"]["change_pct"]["score"]:.1f}%, through a {-T["J2W"]["change_pct"]["final_mse"]:.1f}% lower error. Luna ({-T["Luna"]["change_pct"]["score"]:.1f}%), suliman_tadros ({-T["suliman_tadros"]["change_pct"]["score"]:.1f}%) and we ({-T["mliston"]["change_pct"]["score"]:.1f}%) lowered ours almost entirely through a smaller budget share.'))

S2 = (P('<span class="lead">The estimator never runs the network on sample inputs: it carries a description of the distribution of each layer\'s outputs and updates that description with formulas, one layer at a time.</span> It carries five things.')
      + '<div class="wrap"><ol>'
        '<li>The mean of each of the 1,024 units.</li>'
        '<li>The covariance of every pair of units, a table of 1,024 × 1,024 numbers.</li>'
        '<li>Third-order structure (how lopsided each unit\'s distribution is), stored as a bank of columns: two new columns for every unit that is close to its switching point at each layer, pushed through the weights at every later layer, with the older columns compressed to 384, 320 and 384 directions at layers 7, 10 and 13.</li>'
        '<li>Fourth-order structure (how heavy the tails are) as one number per layer, used for all units alike.</li>'
        '<li>A learned correction: a small network with about 45,000 trained parameters that adjusts each unit\'s mean at each layer.</li>'
        '</ol></div>'
      + fig(diagram(), '<span><i class="k-rect" style="background:transparent;border:2.2px solid var(--s1)"></i>variance input</span><span><i class="k-rect" style="background:transparent;border:2.2px solid var(--s2)"></i>skew input</span><span><i class="k-rect" style="background:transparent;border:2.2px solid var(--s3)"></i>kurtosis input</span>',
            'One layer step. The three inputs with coloured outlines are the ones whose errors sections 4 to 6 measure; the same colours are used in the chart of section 4. The covariance, the bank and the fourth-order number are updated in the same step by matching formulas for pairs of units.')
      + P('Of the compute bill, 77% goes to the third-order bank (43% for pushing the columns through the weights, 34% for reading pair values out of them), 11% to the covariance and the third-order remainder, 5% to the final-layer skew and the pair formulas, and 3% to the learned parts (counted on a slightly earlier version of the estimator).'))

S3 = (P(f'<span class="lead">The error is zero at the first layer and grows by about {slope:.1f}e-9 per layer, to {mean40[15]:.1f}e-9 at the last layer.</span> '
        f'This is the estimator that is on the board, run on 40 validation networks that were not used for training, measured against the organizers\' reference values, which they computed from 1e9 sample inputs per network. '
        f'The step from one layer to the next grows from {steps[2]:.1f}e-9 at layer 3 to {steps[14]:.1f}e-9 at layer 15. The 40 networks end between {nat["lo"][15] * 1e9:.1f}e-9 and {nat["hi"][15] * 1e9:.1f}e-9; their mean, {mean40[15] / 10:.2f}e-8, matches the {sci(us["final_mse"])} of the 50 public board networks.')
      + fig(chart_layers(), '<span><i class="k-thin" style="border-color:var(--s1)"></i>each of 40 validation networks</span><span><i class="k-line" style="border-color:var(--s1)"></i>mean of the 40</span><span><i class="k-line" style="border-color:var(--s2)"></i>noise floor of the organizers\' truth</span>',
            f'Per-layer error of the estimator on the board (entry #{us["id"]}). The noise floor is the mean variance of a unit\'s output divided by 1e9, computed on networks 35, 100, 37 and 39. Only layer 15 counts for the score.', layer_tab, 'Table of the plotted values (×1e-9)')
      + P(f'At layer 0 the estimator is exact, because the input is Gaussian; the {mean40[0]:.2f}e-9 measured there is the sampling noise of the reference values. That noise falls to {floor[15]:.2f}e-9 at layer 15 because the unit outputs get smaller with depth, so from layer 5 on more than 90% of the measured error is real.')
      + P(f'Each layer\'s error has two parts. About {ret_mean * 100:.0f}% of the previous layer\'s error is passed on ({min(ret) * 100:.0f}% to {max(ret) * 100:.0f}% at layers 7 to 15), and new error is added: {rows3[7]["fresh"] * 1e9:.1f}e-9 at layer 7, rising to {r15["fresh"] * 1e9:.1f}e-9 at layer 15. '
          f'The learned correction then removes {min(lrn) * 100:.0f}% to {max(lrn) * 100:.0f}% of the total at each of layers 6 to 15. At layer 15 on the research copy: {r15["carried"] * 1e9:.1f}e-9 carried in, {r15["fresh"] * 1e9:.1f}e-9 new, {r15["host"] * 1e9:.1f}e-9 before the learned correction, {r15["card"] * 1e9:.1f}e-9 after it (the board version on the same four networks: {r15["native"] * 1e9:.1f}e-9). '
          f'Because {ret_mean * 100:.0f}% is passed on at each step, an error made at layer 10 still counts about {ret_mean ** 5 * 100:.0f}% at layer 15, and one made at layer 5 about {ret_mean ** 10 * 100:.0f}%.')
      + f'<div class="wrap"><div class="note">These per-layer parts were measured on the research copy of the estimator, the version used for training, on networks 35, 100, 37 and 39, because the board version does not expose its per-unit inputs. The research copy leaves out a few cost-saving approximations of the board version: its final-layer error is {gap_lo:.1f}% to {gap_hi:.1f}% lower than the board version\'s on networks 39, 37 and 35 and {gap_100:.1f}% higher on network 100, with the same per-unit error pattern (correlation {par_corr[0]:.2f}). Every error number that does not need those inputs is from the board version. Networks 35 and 100 were used for training the learned parts; 37 and 39 were not, and the four give the same results.</div></div>')

S4 = (P('<span class="lead">The new error at each layer comes from three per-unit inputs of one formula, the variance, the skew and the kurtosis of each unit\'s input; the formula itself is accurate.</span> '
        'The mean output of a ReLU unit depends on four properties of its input: the mean <var>μ</var>, the standard deviation <var>σ</var>, the skew (lopsidedness) and the excess kurtosis (tail weight). With <var>a</var> = <var>μ</var>/<var>σ</var>, <var>φ</var> the bell curve and <var>Φ</var> its running total, the estimator uses')
      + '<div class="wrap"><div class="eq">mean after ReLU = <var>σ</var>·[<var>a</var>·<var>Φ</var>(<var>a</var>) + <var>φ</var>(<var>a</var>)] − <var>σ</var>·skew·<var>a</var>·<var>φ</var>(<var>a</var>)/6 + <var>σ</var>·kurtosis·(<var>a</var>² − 1)·<var>φ</var>(<var>a</var>)/24</div></div>'
      + P(f'Fed with the true four inputs, this formula is wrong by less than {max(ft[3:]):.1f}e-9 at every layer from 4 to 15 (networks 35 and 100; the true inputs come from two independent runs of 32 million samples each, so their sampling noise cancels in the comparison). '
          f'Fed with the estimator\'s own inputs, each of layers 10 to 15 adds {rng("var")}e-9 through the variance, {rng("skew")}e-9 through the skew and {rng("kurt")}e-9 through the kurtosis (mean of four networks). The three partly cancel, so the combined new error is {rng("fresh")}e-9, about {(1 - fresh_ratio) * 100:.0f}% less than their sum.')
      + fig(chart_inject(), '<span><i class="k-rect" style="background:var(--s1)"></i>wrong variance</span><span><i class="k-rect" style="background:var(--s2)"></i>wrong skew</span><span><i class="k-rect" style="background:var(--s3)"></i>wrong kurtosis</span>'
            '<span><i class="k-line" style="border-color:var(--ink);border-top-width:2px"></i>the three combined (they partly cancel)</span><span><i class="k-rect" style="background:var(--s1);opacity:.33"></i>pale: not above the noise of the true values</span>',
            'New error added at each layer, by input, mean of networks 35, 100, 37 and 39 (research copy of the estimator). Each value is the mean over the 1,024 units of the squared change of the unit mean caused by that input\'s error. True inputs: 32 million samples per network (16 million for networks 37 and 39); their noise is subtracted, and a segment is drawn pale where the measured value is less than twice that noise.',
            f'<div class="tscroll">{inj_tab}</div><p class="dim sans" style="font-size:12px">All error columns in units of 1e-9. * = not above the noise of the true values.</p>', 'Table: the error budget of every layer')
      + P(f'<b>Variance.</b> Each unit\'s variance is off by {vlo:.2f}% to {vhi:.2f}% at layers 10 to 15 (root-mean-square over units; the noise of the true values is 0.025%). The variance is read from the carried covariance table, and that table\'s error builds up with depth: on network 35 its off-diagonal part is off by {L6["off"]["real"] * 100:.1f}% after layer 6, {L10["off"]["real"] * 100:.1f}% after layer 10 and {L13["off"]["real"] * 100:.1f}% after layer 13. '
          f'About a third of that error at layer 13 ({L13["share_crossval"]["1"] * 100:.0f}%) is a single direction close to the mean vector, but that direction causes little of the per-unit variance error: removing the 16 largest error directions would still leave {L13["next_v"]["unexplained_after_top"]["16"] * 100:.0f}% of it, and removing 64 would leave {L13["next_v"]["unexplained_after_top"]["64"] * 100:.0f}%. The error is spread over hundreds of directions.')
      + P(f'<b>Skew.</b> The estimator\'s per-unit skew follows the true skew closely (correlation 0.998 at layer 15 on networks 35 and 100), and what is missing is {slo:.0f}% to {shi:.0f}% of the unit-to-unit spread of the true skew at layers 10 to 15 ({rows3[1]["g1_err_rel"] * 100:.1f}% at layer 1). The bank that produces this skew is where 77% of the compute goes.')
      + P(f'<b>Kurtosis.</b> The estimator gives every unit of a layer the same kurtosis ({r15["g2_card"]:.3f} at layer 15). The true values average {r15["g2_mean"]:.3f} and differ from unit to unit with a spread of {r15["g2_spread"]:.3f}. At layers 8 to 15 the single value is also {min(klow):.0f}% to {max(klow):.0f}% below the true average; that offset accounts for {min(koff):.0f}% to {max(koff):.0f}% of the kurtosis error and the missing unit-to-unit differences for the rest.')
      + f'<div class="wrap"><details><summary>Table: how wrong each input is, by layer</summary><div class="tscroll">{rel_tab}</div></details></div>'
      + P(f'<b>A direct test.</b> On the research copy, replacing the three inputs by their true values at every layer and switching the learned correction off brings the final-layer error from {a6m("base_learned_off"):.1f}e-9 to {a6m("vsk_learned_off"):.1f}e-9 (mean of four networks; {all_off[0]:.1f}e-9 to {all_off[1]:.1f}e-9 per network). The three inputs therefore account for about {(1 - a6m("vsk_learned_off") / a6m("base_learned_off")) * 100:.0f}% of the error. '
          f'With the learned correction left on, the same replacement gives {a6m("vsk"):.1f}e-9 instead of {b_on:.1f}e-9: the learned correction was trained against the wrong inputs, and with correct inputs it adds error of its own. '
          f'One input at a time, learned correction on: true variances lower the final error by {(1 - a6m("v") / b_on) * 100:.0f}% ({pr("v")} on the four networks), true kurtoses by {(1 - a6m("k") / b_on) * 100:.0f}% ({pr("k")}), true skews by {abs(1 - a6m("s") / b_on) * 100:.0f}% on average ({pr("s")}). The single-input results are small or zero because the three errors partly cancel each other and the learned correction is tuned to all three together.')
      + P(f'<b>The last layer alone.</b> Making the three inputs exact at layer 15 only, with everything before it unchanged, lowers the final-layer error of the research copy by {(1 - f15("vsk") / f15("")) * 100:.0f}% ({f15("") * 1e9:.1f}e-9 to {f15("vsk") * 1e9:.1f}e-9, mean of four networks); the variance alone gives {(1 - f15("v") / f15("")) * 100:.0f}%, the kurtosis {(1 - f15("k") / f15("")) * 100:.0f}% and the skew {(1 - f15("s") / f15("")) * 100:.0f}%. '
          f'For comparison, J2W\'s final-layer error is {(1 - er["J2W"][1]) * 100:.0f}% below ours while its error at the earlier layers equals ours (section 1). The remaining {f15("vsk") / f15("") * 100:.0f}% is error carried in from the earlier layers.')
      + f'<div class="wrap"><div class="tscroll">{ora_tab}</div><p class="dim sans" style="font-size:12px">Final-layer error on the research copy, mean of networks 35, 100, 37 and 39; changes are against the research copy as it is ({b_on:.1f}e-9; the board version on the same networks: {r15["native"] * 1e9:.1f}e-9). The test cannot be run on the board version, which does not expose these inputs. Only the input of the mean formula is replaced; the covariance and the third-order bank keep their own values. The true variances carry sampling noise, which adds 0.6 to 1.4e-9 to the runs that use them; that amount is removed by comparing runs that use one and two independent sets of true values.</p></div>')

g1a, g1w = e1m(15, 'g1_spread'), e1m(15, 'g1_within_spread'); rem = e1r(15, 'skew_var_removed'); kw = e1r(15, 'g2_within_mean')
S5 = (P('<span class="lead">Most of the skew and most of the average kurtosis have one common cause: for a given input the whole previous layer is a little larger or smaller than usual, and that scales the fluctuation of every unit of the next layer up or down together.</span> This shared scaling is called the collective gain mode.')
      + P(f'The measurement: run 16 million sample inputs through each of four networks, sort the inputs into 16 groups by the length of the previous layer\'s output vector, and compute each unit\'s skew and kurtosis inside the groups. '
          f'At layer 15 the average excess kurtosis is {e1m(15, "g2_mean"):.3f} over all inputs and {z(e1m(15, "g2_within_mean"))} inside the groups (between {z(kw[0], 3, True).replace("-", chr(8722))} and {z(kw[1], 3, True)} on the four networks). '
          f'The unit-to-unit spread of the skew is {g1a:.3f} over all inputs and {g1w:.3f} inside the groups, so grouping removes {rem[0] * 100:.0f}% to {rem[1] * 100:.0f}% of the skew variance.')
      + fig(chart_gain(), '<span><i class="k-line" style="border-color:var(--s1)"></i>over all inputs</span><span><i class="k-line" style="border-color:var(--s2)"></i>inside groups of inputs with equal previous-layer norm</span>',
            'Skew and excess kurtosis of the unit inputs at each layer, mean of networks 35, 100, 37 and 39, 16 million samples per network, 16 groups. The sampling noise of one unit is 0.0006 for the skew and 0.0012 for the kurtosis and is subtracted from the spreads. The three panels have different vertical scales.', e1_tab, 'Table of the plotted values')
      + P(f'The zero for the kurtosis overstates the effect. Holding the length of a vector fixed lowers the excess kurtosis of any projection of it by itself, by about 6 divided by the width (0.006 here), and that is what the groups show at layer 0, where there is nothing else: {e1m(0, "g2_within_mean"):.4f}. '
          f'So the {z(e1m(15, "g2_within_mean"))} at layer 15 is a remaining positive kurtosis cancelled by this constraint. A cleaner removal of the shared scaling, dividing each sample by its projection on the mean vector, leaves an average excess kurtosis of 0.012 to 0.019 at layers 6 to 15 (network 35, 16 million samples), against 0.033 to 0.061 without it. '
          f'The shared scaling therefore accounts for about two thirds to four fifths of the average kurtosis, not all of it.')
      + P(f'What grouping does not remove is specific to each network. The unit-to-unit spread of the kurtosis is {e1m(15, "g2_spread"):.3f} over all inputs and still {e1m(15, "g2_within_spread"):.3f} inside the groups, and a skew spread of {g1w:.3f} remains.')
      + P('The estimator\'s errors are in that network-specific part. Its skew error cannot be explained as a wrong size of the shared part or of the rest: a fit of the error on both explains at most 8% of it with equal weight on every unit and at most 17% with the weights that matter for the mean (four networks, layers 1 to 15). '
          'The estimator carries a separate vector for the shared part; it holds 35% to 48% of that part at layers 10 to 15, and the column bank holds the remainder.'))

S6 = (P(f'<span class="lead">Even the true kurtosis of every unit at one layer predicts at most {e4max * 100:.0f}% of the unit-to-unit differences in kurtosis at the next layer, and for skew less than {max(e4skew * 100, 2):.0f}%.</span> '
        f'So storing one more number per unit cannot carry what is missing: the information is in fourth-order relations between pairs of units, and carrying those is estimated to add 50% to 80% to the compute bill (an earlier team estimate, not re-measured today).')
      + P(f'The test used networks 35 and 100 with true values from 32 million samples. The true per-unit values of one layer were pushed through the next layer\'s weights in four ways (by each upstream unit\'s share of the variance, with and without the ReLU\'s response, by squared-weight shares and by absolute shares), and the true values of the next layer were fitted with all of them together. '
          f'The fit explains {e4min * 100:.0f}% to {e4max * 100:.0f}% (R² {e4min:.2f} to {e4max:.2f}) across layers 2 to 15. A second independent 32-million-sample estimate of the target correlates {e4rel[0]:.2f} to {e4rel[1]:.3f} with the first, so sampling noise is not what limits the fit.')
      + f'<div class="wrap"><details><summary>Table: fit quality for every pair of layers</summary><div class="tscroll">{e4_tab}</div></details></div>')

S7 = (P('<span class="lead">Two things are running: long training of the 45,000 learned parameters, which lowers the error at the same budget, and a reduction of the compute bill toward the 10% minimum. Neither reaches 1e-9, and no route to 1e-9 was found today.</span>')
      + f'<div class="wrap"><div class="tscroll">{sc_tab}</div></div>'
      + P(f'<b>Training.</b> The learned correction now removes {(1 - b_on / a6m("base_learned_off")) * 100:.0f}% of the final-layer error (research copy, four networks: {a6m("base_learned_off"):.1f}e-9 without it, {b_on:.1f}e-9 with it). Our earlier trainings ran a few hundred optimiser steps each. One gradient on one network takes 26 seconds on one processor core, so thousands of steps per day are possible, on 640 training networks. '
          f'marius_binner\'s {-mar["change_pct"]["final_mse"]:.1f}% lower error in 72 hours at a nearly unchanged budget share, in {mar["graded_in_window"]} small steps, is what long training of learned parameters would look like; we do not know its method.')
      + P(f'<b>Bill.</b> Our budget share is {us["share"] * 100:.1f}%. At the 10% minimum, with today\'s error, the score would be {sci(us["final_mse"] * 0.1)}, which would be {place(us["final_mse"] * 0.1)} place today. The current target is a bill cut worth 3.5% at unchanged error, which reaches {sci(us["score"] * 0.965)}: below 2.0e-9, and still {place(us["score"] * 0.965)} place.')
      + P(f'<b>What 1e-9 needs.</b> At the 10% minimum it needs a final-layer error of 1.0e-8: {(1 - 1e-8 / us["final_mse"]) * 100:.0f}% below ours and {(1 - 1e-8 / best_err) * 100:.0f}% below the best error on the board. The measurements above say how much that is. True variances alone, with no retraining, would give −{(1 - a6m("v") / b_on) * 100:.0f}%; true variances and kurtoses together −{(1 - a6m("vk") / b_on) * 100:.0f}%. So 1e-9 requires removing most of two of the three input errors, or all three by about half. '
          f'We have no affordable way to do that. The variance error is spread over hundreds of directions of the covariance table. The skew error is {slo:.0f}% to {shi:.0f}% of a quantity that already takes 77% of the bill. The per-unit kurtosis needs carriers for pairs of units. '
          f'Sampling the network does not help at this accuracy: the 10% minimum budget pays for about {round(mc_n, -2):,.0f} full forward passes, which estimate a unit mean with a mean-squared error of about {sci(mc_mse, 0)}, about {round(mc_mse / us["final_mse"], -2):.0f} times our {sci(us["final_mse"], 2)}.')
      + P('What is left to try is therefore training (how far it goes is not known until it has run) and the bill. If training gave what marius_binner got in three days and the bill reached the minimum, the score would be about '
          f'{sci(us["final_mse"] * (1 + mar["change_pct"]["final_mse"] / 100) * 0.1)}: {place(us["final_mse"] * (1 + mar["change_pct"]["final_mse"] / 100) * 0.1)} place today, and still {(us["final_mse"] * (1 + mar["change_pct"]["final_mse"] / 100) * 0.1 / 1e-9 - 1) * 100:.0f}% above 1e-9.'))

page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>47 · Where the remaining error comes from</title>
<style>{CSS}</style>
</head>
<body>
<header class="wrap">
  <span class="eyebrow">ARC white-box estimation · explainer 47 · measured 1 October 2026</span>
  <h1>Where the remaining error comes from</h1>
  <p class="dek">Our final-layer error is {sci(us["final_mse"])}. Almost none of it is made at the last layer: each layer from 5 to 15 adds {rows3[5]["fresh"] * 1e9:.1f} to {rows3[15]["fresh"] * 1e9:.1f}e-9 of new error through three per-unit inputs of one formula, and about {ret_mean * 100:.0f}% of every layer's error is passed on to the next.</p>
  <div class="chips">
    <span class="chip">board entry <b>#{us["id"]}</b> · <b>{sci(us["score"])}</b> · 5th place</span>
    <span class="chip">final-layer error <b>{sci(us["final_mse"])}</b></span>
    <span class="chip">budget share <b>{us["share"] * 100:.1f}%</b></span>
    <span class="chip">per-layer profile on <b>40</b> networks</span>
    <span class="chip">input diagnostics on <b>4</b> networks</span>
  </div>
  <p>The competition gives a random network with 16 layers of 1,024 ReLU units and asks for the average output of every unit when the input is Gaussian noise. The score of one network is the mean-squared error of the 1,024 final-layer averages, multiplied by the share of the compute budget used, with a minimum share of 10%. Each entry is run on 100 networks; the board shows the average over the 50 whose scores are public, and lower is better. Every number below says what it is, on which networks it was measured and against what.</p>
</header>
{sec(1, "the board", "Where we are on the board today", S1)}
{sec(2, "the estimator", "What the estimator carries from layer to layer", S2)}
{sec(3, "error by layer", f"The error grows by about {slope:.1f}e-9 per layer", S3)}
{sec(4, "the three inputs", "Three wrong inputs produce the new error", S4)}
{sec(5, "the shared part", "The collective gain mode", S5)}
{sec(6, "the limit of per-unit carriers", "One number per unit cannot carry what is missing", S6)}
{sec(7, "what happens next", "What the team is doing, and what 1e-9 would need", S7)}
<footer class="wrap">
  <b>How the numbers were measured.</b> Board: public submission pages, ids {board["ids"][0]} to {board["ids"][1]}, read {read_at}. Per-layer error: the saved predictions of the estimator on the board, on 40 validation networks, against the organizers' 1e9-sample reference values.
  Input errors: the research copy of the estimator on networks 35, 100, 37, 39 against Monte Carlo estimates of each unit's input variance, skew and kurtosis (32 million samples, two independent runs, for 35 and 100; 16 million for 37 and 39).
  Collective gain mode: 16 million samples per network, four networks. Covariance error: network 35, against a 32-million-sample covariance with its two halves as the noise reference.<br>
  Scripts, tables and raw outputs: <code>pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/</code> (<code>RESULT.md</code> lists every file). This page is generated by <code>a8_build_explainer.py</code> in that folder.
</footer>
<div id="tt" role="status"></div>
<script>{JS}</script>
</body>
</html>
'''
def nowrap_numbers(doc):
    out = []; skip = 0
    for tok in re.split(r'(<[^>]+>)', doc):
        if tok.startswith('<'):
            m = re.match(r'</?\s*([a-zA-Z0-9]+)', tok)
            if m and m.group(1).lower() in ('svg', 'script', 'style', 'title'): skip += -1 if tok.startswith('</') else 1
            out.append(tok)
        else:
            out.append(tok if skip else re.sub(r'(?<![\w.])(\d+(?:\.\d+)?e-?\d+)(?!\w)', r'<span class="nw">\1</span>', tok))
    return ''.join(out)
page = nowrap_numbers(page)
if len(sys.argv) > 2: page = page.replace('<html lang="en">', f'<html lang="en" data-theme="{sys.argv[2]}">')   # preview only: force a theme
OUT.write_text(page)
print(f'wrote {OUT} ({len(page) / 1024:.1f} kB)')
