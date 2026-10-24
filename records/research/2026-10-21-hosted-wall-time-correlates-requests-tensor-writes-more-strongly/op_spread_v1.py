"""Which op kinds / geometry drive the request and put spread between slow and fast ranks of a native32 run. usage: op_spread_v1.py RUN_DIR"""
import json, glob, sys, statistics
rows = {}
for p in sorted(glob.glob(sys.argv[1] + '/part*/rows.jsonl')):
    for line in open(p): r = json.loads(line); rows[r['rank']] = r
slow = [424, 320, 474, 672, 378, 310, 16, 623, 35, 548]; fast = [400, 473, 316, 651, 287, 300, 200, 333, 557, 530]
def mean_ops(ranks, i):
    keys = set().union(*(rows[r]['operations'].keys() for r in ranks))
    return {k: statistics.mean(rows[r]['operations'].get(k, [0, 0])[i] for r in ranks) for k in keys}
cs, cf = mean_ops(slow, 0), mean_ops(fast, 0); fs, ff = mean_ops(slow, 1), mean_ops(fast, 1)
print('op kind            calls_slow calls_fast  diff   | flops_slow(G) flops_fast(G)')
for k in sorted(cs, key=lambda k: -(cs[k] - cf[k])):
    if abs(cs[k] - cf[k]) >= 20 or k in ('matmul', 'put'):
        print('%-18s %9.0f %9.0f %7.0f | %8.2f %8.2f' % (k, cs[k], cf[k], cs[k] - cf[k], fs[k] / 1e9, ff[k] / 1e9))
print('total calls slow %.0f fast %.0f; requests slow %.0f fast %.0f; puts slow %.0f fast %.0f; put_f4 GB slow %.1f fast %.1f' % (
    sum(cs.values()), sum(cf.values()), statistics.mean(rows[r]['requests'] for r in slow), statistics.mean(rows[r]['requests'] for r in fast),
    statistics.mean(rows[r]['puts'] for r in slow), statistics.mean(rows[r]['puts'] for r in fast),
    statistics.mean(rows[r]['put_bytes']['<f4'] for r in slow) / 1e9, statistics.mean(rows[r]['put_bytes']['<f4'] for r in fast) / 1e9))
# product geometry: count rows by function name
def geo(ranks):
    out = {}
    for r in ranks:
        for g in rows[r]['product_geometry_counts']:
            fn = g[2]; out[fn] = out.get(fn, 0) + g[3] / len(ranks)
    return out
gs, gf = geo(slow), geo(fast)
print('product geometry calls by function (slow vs fast):')
for k in sorted(gs, key=lambda k: -(gs[k] - gf.get(k, 0)))[:12]: print('  %-30s %9.0f %9.0f %+8.0f' % (k, gs[k], gf.get(k, 0), gs[k] - gf.get(k, 0)))
