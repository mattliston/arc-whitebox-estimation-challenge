"""TAG opus-release: one graded run network by network, to say what its worst network was. Platform record only.
Per network: residual, predict wall, kernel seconds per Gflop against the run's median (the speed of the worker while it ran
that network) and whether it is a worker's first network after a setup (it alone carries participant_setup_wall_s; its setup
is the one with the same wall_s, which names the worker and the flops that setup booked). The platform does not publish the
worker of the other networks (a rebuild from the queue order failed its own check on #333559 and #333528: not used).
usage: platform_networks.py <eval json> [<eval json> ...]"""
import json
import statistics as st
import sys


def load(path):
    z = json.load(open(path)); ev = z.get('evaluation') or z; res = ev['results']; nets = []
    for m in res['per_mlp']:
        t = m['telemetry']
        nets.append(dict(i=m['mlp_index'], r=t['residual_wall_time_s'], wall=t['participant_predict_wall_s'], rt=t['participant_roundtrip_wall_s'],
                         k=t['flopscope_server_kernel_wall_s'], o=t['flopscope_server_overhead_wall_s'], flops=t['flops_used'],
                         setup=t.get('participant_setup_wall_s'), lost=m.get('error_code') is not None))
    return ev.get('submission_id'), nets, res.get('setup_executions') or []


for path in sys.argv[1:]:
    sid, nets, setups = load(path); n = len(nets)
    kpf = [x['k']/x['flops']*1e9 for x in nets]; kmed = st.median(kpf)
    for x, v in zip(nets, kpf): x['speed'] = v/kmed
    first = [x for x in nets if x['setup'] is not None]; other = [x for x in nets if x['setup'] is None]
    rmean = st.mean(x['r'] for x in nets); omed = st.median(x['r'] for x in other)
    of = {round(q['wall_s'], 9): q for q in setups}
    print(f"#{sid}: {n} networks, {len(setups)} setups, residual mean {rmean:.4f} s, worst {max(x['r'] for x in nets):.4f} "
          f"(x{max(x['r'] for x in nets)/rmean:.3f} of the mean); wall mean {st.mean(x['wall'] for x in nets):.2f} s, worst {max(x['wall'] for x in nets):.2f}")
    print(f"  first networks ({len(first)}): residual mean {st.mean(x['r'] for x in first):.4f}, max {max(x['r'] for x in first):.4f}; "
          f"other ({len(other)}): mean {st.mean(x['r'] for x in other):.4f}, median {omed:.4f}, max {max(x['r'] for x in other):.4f}; "
          f"first / other (means) x{st.mean(x['r'] for x in first)/st.mean(x['r'] for x in other):.3f}")
    xs = [x['speed'] for x in other]; ys = [x['r'] for x in other]; mx, my = st.mean(xs), st.mean(ys)
    cov = sum((a - mx)*(b - my) for a, b in zip(xs, ys)); corr = cov/(sum((a - mx)**2 for a in xs)*sum((b - my)**2 for b in ys))**0.5
    slope = cov/sum((a - mx)**2 for a in xs)
    print(f"  other networks: correlation of residual with kernel seconds per flop {corr:.2f}; residual rises {slope/my*0.01*100:.2f}% per 1% of "
          f"kernel time per flop; kernel per flop spread: 10th-90th percentile x{sorted(xs)[len(xs)//10]:.3f}-x{sorted(xs)[-len(xs)//10]:.3f}, largest x{max(xs):.3f}")
    print(f"  setups: " + '; '.join(f"{s['worker_id']} {s['flops']/1e9:.3f} G flops {s['wall_s']:.3f} s" for s in setups))
    print("  the eight largest residuals (network, residual, / median of the other networks, kernel per flop / run median, wall; for a first network its setup):")
    for x in sorted(nets, key=lambda x: -x['r'])[:8]:
        q = of[round(x['setup'], 9)] if x['setup'] is not None else None
        print(f"    {x['i']:3d}  {x['r']:.4f} s  x{x['r']/omed:.3f}  x{x['speed']:.3f}  {x['wall']:.2f} s" + (f"  first network of {q['worker_id']}, setup {q['wall_s']:.3f} s, {q['flops']:,} flops" if q else ''))
    print("  first networks in order (network, worker, setup flops, residual / median of the others, kernel per flop / run median):")
    for x in sorted(first, key=lambda x: x['i']):
        q = of[round(x['setup'], 9)]
        print(f"    {x['i']:3d}  {q['worker_id']}  {q['flops']:,}  x{x['r']/omed:.3f}  x{x['speed']:.3f}")
    print()
