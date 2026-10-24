"""dissect337259 agent 2026-10-20: re-weight the record's PRE-weight charge census (probe_charges_v2 output) with flopscope 0.12.3
default weights and aggregate by op / layer / function / site. usage: census_reweight_v1.py <charges json> <out json>"""
import collections, json, sys
d = json.load(open(sys.argv[1]))
W = {'split': 0.0, 'transpose': 0.0, 'unstack': 0.0, 'take': 4.0, 'argsort': 4.0, 'exp': 16.0}
tot = 0; by_site = collections.defaultdict(float); by_fn = collections.defaultdict(float); by_layer = collections.defaultdict(float); by_op = collections.defaultdict(float); by_fn_layer = collections.defaultdict(float)
for op, f, sub, sh, dt, st, lay in d['rows']:
    w = W.get(op, 1.0) * (2.0 if any(x == 'float64' for x in dt) else 1.0)
    c = f * w; tot += c
    by_site[(op, st)] += c; by_fn[(op, st.split(':')[0])] += c; by_layer[lay] += c; by_op[op] += c; by_fn_layer[(st.split(':')[0], lay)] += c
print('weighted total %.3f GF (pre-weight %.3f)' % (tot / 1e9, d['total_flops'] / 1e9))
for name, table in (('op', by_op), ('layer', by_layer)):
    print('--- by', name)
    for k, v in sorted(table.items(), key=lambda t: (-t[1]) if name == 'op' else (t[0] is None, t[0])): print('  %-24s %8.3f GF %6.2f%%' % (k, v / 1e9, 100 * v / tot))
print('--- by function'); [print('  %-14s %-36s %8.3f GF %6.2f%%' % (k[0], k[1], v / 1e9, 100 * v / tot)) for k, v in sorted(by_fn.items(), key=lambda t: -t[1])[:40]]
print('--- top sites'); [print('  %-14s %-40s %8.3f GF %6.2f%%' % (k[0], k[1], v / 1e9, 100 * v / tot)) for k, v in sorted(by_site.items(), key=lambda t: -t[1])[:40]]
json.dump(dict(total=tot, by_op=dict(by_op), by_fn={'%s|%s' % k: v for k, v in by_fn.items()}, by_layer={str(k): v for k, v in by_layer.items()},
               by_fn_layer={'%s|%s' % k: v for k, v in by_fn_layer.items()}, by_site={'%s|%s' % k: v for k, v in by_site.items()}), open(sys.argv[2], 'x'))
