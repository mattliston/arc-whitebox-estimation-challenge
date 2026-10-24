"""[opus-lead] paired per-network telemetry ratios of two graded/partial evaluations (same networks): residual, kernel, overhead, comms,
predict wall, flops; split cold (setup_generation 0) / warm. usage: paired_telemetry_v1.py <eval_new.json> <eval_old.json>"""
import json, sys, statistics as st
def per(p):
    ev = json.load(open(p))['evaluation']; return {m.get('mlp_index'): m for m in ev['results']['per_mlp'] if m.get('telemetry')}
a, b = per(sys.argv[1]), per(sys.argv[2]); ks = sorted(set(a) & set(b))
F = [('residual', 'residual_wall_time_s'), ('kernel', 'flopscope_server_kernel_wall_s'), ('overhead', 'flopscope_server_overhead_wall_s'),
     ('comms', 'flopscope_server_comms_wall_s'), ('predict', 'participant_predict_wall_s'), ('flops', 'flops_used')]
def ratios(keys):
    out = {}
    for name, f in F:
        r = [a[k]['telemetry'][f] / b[k]['telemetry'][f] for k in keys if a[k]['telemetry'].get(f) and b[k]['telemetry'].get(f)]
        if r: out[name] = (st.median(r), st.mean(r), max(r), len(r))
    return out
cold = [k for k in ks if (a[k]['telemetry'].get('setup_generation') or 0) == 0 and (b[k]['telemetry'].get('setup_generation') or 0) == 0]
warm = [k for k in ks if k not in cold]
for label, keys in (('ALL', ks), ('cold', cold), ('warm', warm)):
    if not keys: continue
    r = ratios(keys)
    print(label, len(keys), ' | '.join('%s med %.3f mean %.3f max %.3f' % (n, v[0], v[1], v[2]) for n, v in r.items()))
new_res = [a[k]['telemetry']['residual_wall_time_s'] for k in ks]; old_res = [b[k]['telemetry']['residual_wall_time_s'] for k in ks]
print('residual seconds on these nets: new median %.4f max %.4f | old median %.4f max %.4f' % (st.median(new_res), max(new_res), st.median(old_res), max(old_res)))
print('cold nets (new):', [(k, round(a[k]['telemetry']['residual_wall_time_s'], 3)) for k in cold])
