"""Per-network wall anatomy of a native32 run dir: correlates wall with FLOPs, requests, pushes, Smirnov calls. usage: wall_anatomy_v1.py RUN_DIR"""
import json, glob, math, statistics, sys
rows = []
for p in sorted(glob.glob(sys.argv[1] + '/part*/rows.jsonl')):
    for line in open(p): rows.append(json.loads(line))
rows.sort(key=lambda r: -r['wall_s'])
print('rank    wall backend flops(G)   req   puts  putGB sm40 sm40rowsM  peak_h')
for r in rows:
    sm = r['_ASTRA_SM40_STATS']; rowsum = sum(s[0] * s[1] for s in sm['shapes'])
    print('%4d %7.1f %7.1f %7.1f %6d %6d %6.1f %2d %8.1f %6d' % (r['rank'], r['wall_s'], r['backend_s'], r['flops'] / 1e9, r['requests'], r['puts'], r['put_bytes']['<f4'] / 1e9, sm['calls'], rowsum / 1e6, r['peak_handles']))
def corr(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
W = [r['wall_s'] for r in rows]
for name, f in [('flops', lambda r: r['flops']), ('requests', lambda r: r['requests']), ('puts', lambda r: r['puts']), ('put_f4', lambda r: r['put_bytes']['<f4']),
                ('sm40calls', lambda r: r['_ASTRA_SM40_STATS']['calls']), ('sm40rows', lambda r: sum(s[0] * s[1] for s in r['_ASTRA_SM40_STATS']['shapes'])), ('peak_handles', lambda r: r['peak_handles'])]:
    print('corr(wall,%s)=%.3f' % (name, corr(W, [f(r) for r in rows])))
r = rows[0]; ops = r['operations']
print('operations type', type(ops).__name__, 'len', len(ops) if hasattr(ops, '__len__') else '')
if isinstance(ops, dict):
    items = sorted(ops.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else (kv[1].get('flops', 0) if isinstance(kv[1], dict) else 0)))
    for k, v in items[:25]: print('  ', k, str(v)[:160])
else: print(str(ops)[:1500])
print('product_geometry_counts', str(r['product_geometry_counts'])[:800])
