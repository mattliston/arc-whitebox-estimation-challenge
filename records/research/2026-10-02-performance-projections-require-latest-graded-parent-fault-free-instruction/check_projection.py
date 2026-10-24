"""TAG opus-release: after a fire, compare what the platform measured with what stage C projected, part by part (kernel, overhead,
residual, predict wall, worst network, setup), and say how far each was off. The platform record is the evaluation JSON the
lead or the analyst stores (the same shape as H/eval_333111.json: evaluation.results.per_mlp[].telemetry and setup_executions).
Also prints the per-request and per-round-trip constants the platform implies, given the local request counts, and what every
setup booked (a setup that booked fewer flops than the others did not bind every key before its guard).
usage: check_projection.py <eval_<id>.json> <runs/<tag>/stage_c.json>"""
import json
import statistics as st
import sys

record = json.load(open(sys.argv[1]))
stage = json.load(open(sys.argv[2]))
results = record['evaluation']['results'] if 'evaluation' in record else record['results']
tel = [m['telemetry'] for m in results['per_mlp'] if m.get('telemetry')]
lost = [(m['mlp_index'], m.get('error_code')) for m in results['per_mlp'] if m.get('error_code') not in (None, 'SUCCESS')]
col = lambda k: [t[k] for t in tel if t.get(k) is not None]
p, c = stage['projection'], stage['candidate_summary']
actual = dict(kernel=st.mean(col('flopscope_server_kernel_wall_s')), overhead=st.mean(col('flopscope_server_overhead_wall_s')),
              residual=st.mean(col('residual_wall_time_s')), wall=st.mean(col('participant_predict_wall_s')),
              wall_max=max(col('participant_predict_wall_s')), residual_max=max(col('residual_wall_time_s')),
              flops=st.mean(col('flops_used')), comms=st.mean(col('flopscope_server_comms_wall_s')))
projected = dict(kernel=p['kernel_s'], overhead=p['overhead_s'], residual=p['residual_mean_s'], wall=p['mean_wall_s'],
                 wall_max=p['worst_wall_s'], residual_max=p['residual_max_s'], flops=c['flops'])
print(f"{sys.argv[1]}: {len(tel)} networks with telemetry, lost: {lost or 'none'}")
print(f"{'quantity':28s} {'projected':>10s} {'platform':>10s} {'error':>8s}")
names = dict(kernel='kernel, mean s', overhead='overhead, mean s', residual='residual, mean s', wall='predict wall, mean s',
             wall_max='predict wall, worst network s', residual_max='residual, worst network s',
             flops='bill per network (local 35,100)')
for k, label in names.items():
    a, b = projected[k], actual[k]
    print(f"{label:28s} {a:10.4g} {b:10.4g} {100*(a/b - 1):+7.1f}%")
print(f"kernel band projected {p['kernel_lo_s']:.1f}-{p['kernel_hi_s']:.1f} s; worst-network wall band "
      f"{p['worst_wall_lo_s']:.1f}-{p['worst_wall_hi_s']:.1f} s")
print(f"platform constants with the local counts ({c['requests']:.0f} requests, {c['round_trips']:.0f} round trips per network): "
      f"residual {1e6*actual['residual']/c['requests']:.2f} microseconds per request, overhead "
      f"{1e3*actual['overhead']/c['round_trips']:.4f} ms per round trip, wire time {1e6*actual['comms']/c['round_trips']:.1f} "
      f"microseconds per round trip")
print(f"a network may take x{120/actual['wall']:.2f} the mean wall and x{0.4/actual['residual']:.2f} the mean residual; the worst "
      f"network took x{actual['wall_max']/actual['wall']:.3f} and x{actual['residual_max']/actual['residual']:.3f}")
print('setups (worker, flops booked, wall s, kernel ms, wire ms):')
for s in results.get('setup_executions', []):
    print(f"  {s['worker_id']} gen {s.get('generation')}: {s['flops']:,} flops, {s['wall_s']:.3f} s, {s['kernel_ns']/1e6:.0f} ms, "
          f"{s['comms_ns']/1e6:.0f} ms, {s.get('error_code')}")
