"""[opus-pass-release] first live run of the governor (#336506 pubmaxapg2 = swap + uncapped read grid + public-fire thresholds):
per-network telemetry paired by mlp_name with #336412 (S2tr, no governor, same 100 networks): flops ratio, residual and overhead
ratios (both scale with round trips/requests), kernel and wall ratios; a governor tier is inferred where the network departs from
the run's median ratios (EXACT: flops up ~+1.7% and kernel down; FULL/EXACT residual tiers: residual and overhead down 8-20%).
Public vs hidden from the evaluation's public_scores list. usage: analyze_governor_live_v1.py <eval_new.json> <eval_ref.json>"""
import json, statistics as st, sys
new, ref = (json.load(open(p))['evaluation'] for p in sys.argv[1:3])
def nets(e):
    return {m['mlp_name']: m for m in e['results']['per_mlp']}
N, R = nets(new), nets(ref)
pub = set()
for k in ('public_scores',):
    v = new['results'].get(k)
    if isinstance(v, list):
        for x in v:
            if isinstance(x, dict) and x.get('mlp_name'):
                pub.add(x['mlp_name'])
            elif isinstance(x, dict) and 'mlp_index' in x:
                pub |= {m['mlp_name'] for m in new['results']['per_mlp'] if m['mlp_index'] == x['mlp_index']}
    elif isinstance(v, dict):
        pub |= set(v.keys())
rows = []
for name, m in N.items():
    t = m.get('telemetry') or {}; r = (R.get(name) or {}).get('telemetry') or {}
    rows.append(dict(name=name, idx=m['mlp_index'], public=name in pub, state=m.get('state'), err=m.get('error_code'),
                     flops=t.get('flops_used'), wall=t.get('participant_predict_wall_s'), resid=t.get('residual_wall_time_s'),
                     over=t.get('flopscope_server_overhead_wall_s'), kern=t.get('flopscope_server_kernel_wall_s'), setup=t.get('participant_setup_wall_s'),
                     rflops=r.get('flops_used'), rwall=r.get('participant_predict_wall_s'), rresid=r.get('residual_wall_time_s'),
                     rover=r.get('flopscope_server_overhead_wall_s'), rkern=r.get('flopscope_server_kernel_wall_s')))
ok = [x for x in rows if x['flops'] and x['rflops'] and x['resid'] and x['rresid']]
ratio = lambda x, a, b: x[a] / x[b]
med = {k: st.median(ratio(x, a, b) for x in ok) for k, a, b in (('flops', 'flops', 'rflops'), ('resid', 'resid', 'rresid'), ('over', 'over', 'rover'), ('kern', 'kern', 'rkern'), ('wall', 'wall', 'rwall'))}
print('networks', len(rows), 'paired', len(ok), 'public', len(pub), '| median ratios new/ref:', {k: round(v, 4) for k, v in med.items()})
fails = [x for x in rows if x['state'] != 'completed' or x['err'] not in (None, 'SUCCESS')]
print('FAILURES', len(fails), [(x['idx'], x['name'], 'public' if x['public'] else 'hidden', x['err']) for x in fails])
tier = []
for x in ok:
    fr, rr, orr = ratio(x, 'flops', 'rflops') / med['flops'], ratio(x, 'resid', 'rresid') / med['resid'], ratio(x, 'over', 'rover') / med['over']
    kind = []
    if fr > 1.008: kind.append('EXACT?(flops %+.1f%%)' % (100 * (fr - 1)))
    if rr < 0.93 and orr < 0.95: kind.append('residual tier?(resid %+.0f%% over %+.0f%%)' % (100 * (rr - 1), 100 * (orr - 1)))
    if kind:
        tier.append((x['idx'], x['name'], 'public' if x['public'] else 'hidden', round(x['wall'], 1), round(x['resid'], 3), ' '.join(kind)))
print('GOVERNOR-LIKE DEPARTURES', len(tier)); [print('  ', t) for t in tier]
for k in ('wall', 'resid', 'kern', 'over'):
    v = sorted(x[k] for x in rows if x[k]); w = sorted(x['r' + k] for x in rows if x['r' + k])
    print(f'{k:5s} new mean {st.mean(v):.3f} p90 {v[int(.9 * len(v))]:.3f} max {v[-1]:.3f} | ref mean {st.mean(w):.3f} p90 {w[int(.9 * len(w))]:.3f} max {w[-1]:.3f}')
agg = new['results']['aggregates']['public']; ra = ref['results']['aggregates']['public']
print('PUBLIC score new %.6e ref %.6e (%+.3f%%) | MSE %+.3f%% | util %+.3f%% | failed new %s ref %s' % (agg['score'], ra['score'], 100 * (agg['score'] / ra['score'] - 1),
      100 * (agg['final_layer_mse'] / ra['final_layer_mse'] - 1), 100 * (agg['mean_compute_utilization'] / ra['mean_compute_utilization'] - 1), agg['n_failed_mlps'], ra['n_failed_mlps']))
