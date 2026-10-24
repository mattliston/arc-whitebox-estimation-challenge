#!/usr/bin/env python3
"""TAG opus-lead: paired per-network comparison of two graded submissions from their public evaluation JSONs (read_eval_page.py output).
usage: compare_evals.py <eval_new.json> <eval_old.json>"""
import sys, json, math
new, old = (json.load(open(p))['evaluation'] for p in sys.argv[1:3])
def per(ev):
    out = {}
    for m in ev['results']['per_mlp']:
        key = m.get('mlp_index', m.get('index', m.get('mlp_id')))
        out[key] = m
    return out
a, b = per(new), per(old); keys = sorted(set(a) & set(b))
print('fields of one per-network record:', sorted(k for k in a[keys[0]] if k != 'telemetry')[:20])
def g(m, *names):
    for n in names:
        if n in m and m[n] is not None: return m[n]
    return None
rows = []
for k in keys:
    x, y = a[k], b[k]
    rows.append((k, g(x, 'final_layer_mse'), g(y, 'final_layer_mse'), g(x, 'compute_utilization', 'score_multiplier'), g(y, 'compute_utilization', 'score_multiplier'),
                 g(x, 'adjusted_final_layer_score', 'score'), g(y, 'adjusted_final_layer_score', 'score')))
rows = [r for r in rows if all(v is not None for v in r[1:])]
n = len(rows)
if n:
    mse = sum(r[1] for r in rows) / sum(r[2] for r in rows); util = sum(r[3] for r in rows) / sum(r[4] for r in rows); adj = sum(r[5] for r in rows) / sum(r[6] for r in rows)
    print(f'networks compared {n}: pooled final-layer MSE x{mse:.4f} (lower on {sum(r[1] < r[2] for r in rows)}/{n}), budget share x{util:.4f}, adjusted x{adj:.4f} (lower on {sum(r[5] < r[6] for r in rows)}/{n})')
    d = [r[5] / r[6] - 1 for r in rows]; m = sum(d) / n; sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    print(f'per-network adjusted change: mean {100*m:+.2f}% se {100*sd/math.sqrt(n):.2f}%, worst {100*max(d):+.2f}%, best {100*min(d):+.2f}%')
for name, ev in (('new', new), ('old', old)):
    agg = ev['results']['aggregates']['public']; t = [m['telemetry'] for m in ev['results']['per_mlp'] if m.get('telemetry')]
    col = lambda k: sorted(x[k] for x in t if k in x)
    w, r_ = col('participant_predict_wall_s'), col('residual_wall_time_s')
    print(f"{name}: score {agg['score']:.4e}, final-layer MSE {agg['final_layer_mse']:.4e}, all-layer {agg.get('all_layers_mse')}, budget share {agg['mean_compute_utilization']:.5f}, failed {agg['n_failed_mlps']}; "
          f"wall mean {sum(w)/len(w):.1f} max {w[-1]:.1f} s; residual mean {sum(r_)/len(r_):.3f} max {r_[-1]:.3f} s; networks {len(t)}")
