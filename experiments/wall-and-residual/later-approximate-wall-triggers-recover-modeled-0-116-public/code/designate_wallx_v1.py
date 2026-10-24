"""[opus-wallx] governor tier model for GOV v4h (stronger wall tiers) on the swap + A stack. Base = u64's designate_v1 network model
(copied in release's designate_pubmax_v1: platform record rows, FULL/EXACT/LEAN request savings, 3% projection slack, failure at
residual > 0.4 s or wall > 119.6 s). Two changes, both from the code:
 (1) EXACT-AWARE RELIEF: the EXACT set turns _ASTRA_SM40_ON off, and A's 1,335 grid keys are _ASTRA_SM40_PLANS entries, so under EXACT the
     rest of an A network runs the swap's own EXACT paths: relief = rem * (W - 0.88 * W_swap) with W_swap = W * r_swap / r_cand (swap:
     rem * 0.12 W as in designate_v1).
 (2) LEAN PROJECTION: 'raw' = release's model (LEAN when the raw projection > W); 'post' = the code's later-layer projection already
     includes EXACT's speed-up, so LEAN when the EXACT-relieved projection > W (relief applied from the same layer: slightly optimistic).
usage: designate_wallx_v1.py  (writes DESIGNATE_wallx_v1.json)"""
import glob, json, os
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = O/'lead/runs/opus_governor_v1'
rp = json.loads((D/'REPLAY_v4f.json').read_text())
N = 45627
FULL = {l: (309 if l >= 13 else 315) for l in range(1, 15)}
EXACT = {int(k): v for k, v in rp['exact_per_layer'].items()}
LEAN = {**{l: 453 for l in range(3, 8)}, **{l: 561 for l in range(8, 14)}, 14: 514, 15: 234}
WSHARE = {3: 0.092, 6: 0.302, 8: 0.446}
COST = dict(full=1.65, exact=1.65, lean_mse=36.0, shed_mse=3800.0)
saved = lambda t, L: sum(v for l, v in t.items() if l >= L)


def network(R, W, g, L, xratio, mode):
    s, bill, mse = 0, 0.0, 0.0
    if R > g['T']:
        s += saved(FULL, L); bill += COST['full']
    if R > g['T2']:
        s += saved(EXACT, L); bill += COST['exact']
    rem = 1 - WSHARE[L]
    exact_w = W > g['W1']
    Wn = W
    if exact_w:
        Wn -= rem * (W - 0.88 * W * xratio)          # xratio = r_swap / r_cand (1 for the swap)
        if not R > g['T2']:
            bill += COST['exact']
    lean_proj = W if mode == 'raw' else Wn
    lean = R > g['T3'] or lean_proj > g['W']
    if R > g['T3']:
        s += saved(LEAN, L)
    Rn = R * (1 - s / N)
    if lean:
        Wn -= W * rem * 0.031 + 0.22e-3 * saved(LEAN, L)
        mse += COST['lean_mse']
    shed = Wn > g['W2'] * 0.97
    if shed:
        Wn -= W * (1 - WSHARE[max(L, 6)]) * 0.33
        mse += COST['shed_mse']
    return (Rn > 0.4 or Wn > 119.6), bill, mse, Wn, lean, shed


rows = []
for f in sorted(glob.glob(str(O/'fire/*/eval_33*.json')), key=os.path.getmtime)[-12:]:
    d = json.load(open(f)); pm = (d.get('evaluation') or {}).get('results', {}).get('per_mlp') or []
    tel = [m.get('telemetry') or {} for m in pm]
    done = [t['flops_used'] for t, m in zip(tel, pm) if m.get('state') == 'completed' and t.get('flops_used')]
    if not done or not any(t.get('residual_wall_time_s') for t in tel):
        continue
    fmed = sorted(done)[len(done) // 2]
    for t, m in zip(tel, pm):
        if m.get('error_code') == 'TIME_EXHAUSTED':
            rows.append((0.36, 120.0 * fmed / t['flops_used']))
        elif t.get('residual_wall_time_s') is not None and t.get('participant_predict_wall_s') is not None:
            rows.append((t['residual_wall_time_s'], t['participant_predict_wall_s']))
SPEC = {'swap': dict(req=0.99875, wall=1.01478, bill=0.0), 'A': dict(req=1.00298, wall=1.06404, bill=-0.148)}   # release spec; bill = public-mix vs swap
GOVS = {'v4g': dict(T=0.39, T2=0.41, T3=0.46, W1=112.0, W=118.0, W2=123.0)}
for w in (120.0, 121.0, 122.0, 124.0):
    for w2 in (123.0, 125.0, 127.0):
        if w2 > w:
            GOVS[f'v4h_W{w:.0f}_W2{w2:.0f}'] = dict(T=0.39, T2=0.41, T3=0.46, W1=112.0, W=w, W2=w2)
res = {}
print('networks', len(rows))
for mode in ('raw', 'post'):
    for gname, g in GOVS.items():
        for cname, sp in SPEC.items():
            xr = SPEC['swap']['wall'] / sp['wall']
            for L in (3, 6, 8):
                out = [network(R * sp['req'], W * sp['wall'], g, L, xr, mode) for R, W in rows]
                f = sum(o[0] for o in out) / len(rows)
                res[f'{mode}|{gname}|{cname}|L{L}'] = dict(p_public_50=100 * (1 - (1 - f) ** 50), p_run_100=100 * (1 - (1 - f) ** 100),
                    gov_bill=sum(o[1] for o in out) / len(rows), gov_mse=sum(o[2] for o in out) / len(rows), lean=sum(o[4] for o in out),
                    shed=sum(o[5] for o in out), wmax=max(o[3] for o in out), bill=sp['bill'])
base = lambda mode, L: res[f'{mode}|v4g|swap|L{L}']
print('net vs swap+v4g = bill + d(gov bill) + d(gov MSE), in %; P(public)/P(any) %; LEAN/SHED counts per 1,200')
for mode in ('raw', 'post'):
    for gname in GOVS:
        for cname in SPEC:
            line = f'{mode:4s} {gname:16s} {cname:4s}'
            for L in (3, 6, 8):
                r, b = res[f'{mode}|{gname}|{cname}|L{L}'], base(mode, L)
                net = r['bill'] + r['gov_bill'] - b['gov_bill'] + r['gov_mse'] - b['gov_mse']
                r['net_vs_swap_v4g'] = net
                line += f' | L{L} net {net:+.3f} P {r["p_public_50"]:.1f}/{r["p_run_100"]:.1f} lean {r["lean"]} shed {r["shed"]} wmax {r["wmax"]:.1f}'
            print(line)
(Path(__file__).parent/'DESIGNATE_wallx_v1.json').write_text(json.dumps(res, indent=1))
