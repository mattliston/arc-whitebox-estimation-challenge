"""TAG scout-card (2026-10-12): two result JSONs of the same request content (e.g. gk on worker_sc1 + the engineer's cut simulator vs gk on
worker_sc5 + the scout's private simulator copy with its new keywords at 0): BIT FOR BIT or not (mse, mse_layers, final_prediction).
usage: python k_ctl_pair_v1.py <result a.json> <result b.json>"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = {k: a[k] == b[k] for k in ('mse', 'mse_layers', 'final_prediction')}
worst = max(abs(x - y) / max(abs(y), 1e-300) for x, y in zip(a['mse_layers'], b['mse_layers']))
print('BIT FOR BIT' if all(same.values()) else 'DIFFERS', same, 'max rel layer diff %.3g' % worst, 'threads', a['threads'], b['threads'])
