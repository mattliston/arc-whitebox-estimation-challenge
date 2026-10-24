"""[opus-small] per-form summary of the native variant bench: for each shape, per (executable orientation, L, packets) the FLOP and cycle
deltas vs the live-route incumbent; which (ori, L) is FLOP-best per form, and how much the 2% margin + MF/Gcyc rules keep."""
import json, glob, collections, sys
S = sys.argv[1]; rows = {}
for f in glob.glob(S + '/BENCH_small_shard*.json'): rows.update(json.load(open(f)))
def form(s):
    m, k, n = eval(s); return ''.join('R' if x == 384 else 'x' for x in (m, k, n))
agg = collections.defaultdict(lambda: collections.Counter())
for s, r in rows.items():
    inc = r['inc']; fm = form(s)
    for v in r['variants']:
        if 'flops' not in v: continue
        key = (fm, v['ori'], v['L'], v['packets'])
        dF, dC = v['flops'] - inc['flops'], v['cycles'] - inc['cycles']
        agg[key]['n'] += 1; agg[key]['dF'] += dF; agg[key]['dC'] += dC
        if dF <= -0.02 * inc['flops']: agg[key]['win2'] += 1; agg[key]['dF2'] += dF; agg[key]['dC2'] += dC
for key in sorted(agg, key=lambda k: (k[0], agg[k]['dF2'])):
    a = agg[key]
    if a['win2'] == 0: continue
    print(f"form {key[0]} ori {key[1]} L{key[2]} {'pk' if key[3] else '  '} shapes {a['n']:3d} wins>=2% {a['win2']:3d}  dF@2% {a['dF2']/1e6:9.1f}M  dC@2% {a['dC2']/1e6:9.1f}Mc  ratio {(-a['dF2']/1e6)/(a['dC2']/1e9) if a['dC2'] > 0 else float('inf'):7.1f} MF/Gc")
