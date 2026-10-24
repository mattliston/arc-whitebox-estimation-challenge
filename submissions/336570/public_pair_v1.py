"""[opus-solo] paired public-network comparison of two graded (or partially graded) runs from their eval JSONs: per public net
(idx 0-49) final_layer_mse and flops_used, on the nets both completed. Reports mean log ratios (MSE, bill, score = MSE x util),
SE, count lower, and the aggregate (sum MSE x util over the paired nets). usage: public_pair_v1.py <eval_new.json> <eval_ref.json>"""
import json, math, statistics as st, sys
new, ref = (json.load(open(p))['evaluation'] for p in sys.argv[1:3])
def rows(d):
    pm = {p['mlp_index']: p for p in d['results']['per_mlp']}
    sc = {s['mlp_index']: s['scores'] for s in (d['results'].get('public_scores') or [])}
    out = {}
    for i, s in sc.items():
        p = pm.get(i)
        if p and p['state'] == 'completed' and s.get('final_layer_mse'):
            f = p['telemetry']['flops_used']; out[i] = (s['final_layer_mse'], f, s['final_layer_mse'] * max(f / 2**41, 0.1))
    return out
a, b = rows(new), rows(ref)
ks = sorted(set(a) & set(b))
print(f"paired public nets: {len(ks)} (new has {len(a)}, ref has {len(b)}); new state {new['state']} progress {new['progress']['mlps_completed']}/{new['progress']['total_mlps']} failed {new['progress']['mlps_failed']}")
if len(ks) >= 2:
    for j, name in ((0, 'final MSE'), (1, 'bill'), (2, 'score')):
        v = [100 * math.log(a[k][j] / b[k][j]) for k in ks]
        print(f"  {name:9s} mean {st.mean(v):+7.3f}%  se {st.stdev(v) / len(v) ** .5:.3f}  lower {sum(x < 0 for x in v)}/{len(v)}")
    agg = 100 * math.log(sum(a[k][2] for k in ks) / sum(b[k][2] for k in ks))
    print(f"  AGGREGATE score on the paired nets {agg:+.3f}%  (ref aggregate on these nets {sum(b[k][2] for k in ks) / len(ks):.4e})")
