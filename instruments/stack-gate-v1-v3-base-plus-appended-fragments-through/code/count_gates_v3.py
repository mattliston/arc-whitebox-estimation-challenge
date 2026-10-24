"""[opus-lowres] COUNT gates v3 = v2 + the lead's re-fit (2026-10-16 ~14:56Z): projected worst wall <= 106 s (normal judge), modeled
P(fail) on the S2tr-family noise (runs 336381/389/392/397/412) as the headline, all-Oct-16 (pessimistic) and pre-Oct-16 also printed.
v2 = the lead's GATE RESET + correction (2026-10-16 ~14:50Z; supersedes v1's 43,800 / 42,700 / 100 s),
Astra SAFE-build level: on BOTH the normal and the 1.3x-slow boundary, first network (rank 35) <= 42,700 and warm network (rank 100)
<= 41,200 requests; projected worst-network wall (normal-boundary judge) <= 100 s; projected platform setup <= 4.60 s (whole-run and
1.3x-slow-setup judge lines); no FAIL judge line.
Plus: (a) v1's residual projection (every #336412 network's residual x this build's request x residual-instructions-per-request ratio
vs S2tr's boundary, first nets with rank 35's ratio, warm with rank 100's); (b) projected platform run medians: residual = #336412's
median x the warm ratio (and warm requests x 7.77 us), wall = #336412's median round-trip wall x (judge mean wall / S2tr's judge mean);
(c) modeled P(any fail) / P(public fail) per 100-network run under BOTH noise regimes: per-network log offsets (residual and round-trip
wall) vs the run's medians from fire/*/eval_*.json runs since 2026-10-13 = network effect (each net's mean over runs; the same 100 nets
every run) + noise; noise kept as (residual, wall) pairs (they co-move) and split by the run's started_at before / after 2026-10-16T00:00Z;
a network fails if residual > 0.4 s or round-trip wall > 120 s (wall-killed nets read residual 0: wall offset only); nets independent
(public = idx 0-49), and, as a check, a whole-run bootstrap (each past run's own offsets, clustered noise).
Rows from <dir>/boundary/{normal,slow}/rows.jsonl (release dir) or <dir>/{normal,slow}/rows.jsonl (probe); judge lines from every
judge.txt under <dir> (normal = a parent dir named *NORMAL*). Writes COUNT_GATES_v3_<dirname>[_<utc>].json (never overwrites).
usage: count_gates_v3.py <dir> <s2tr_rel_dir> <eval_336412.json>"""
import glob, json, math, re, statistics as st, subprocess, sys
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D, S, EV = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
LIM = {35: 42700, 100: 41200}; WALL, SETUP, RCAP, WCAP, US = 106.0, 4.60, 0.4, 120.0, 7.77e-6; CUT = '2026-10-16T00:00:00Z'; SINCE = '2026-10-13T00:00:00Z'; FAMILY = {'336381', '336389', '336392', '336397', '336412'}


def rows(d, m):
    p = d/'boundary'/m/'rows.jsonl'
    p = p if p.exists() else d/m/'rows.jsonl'
    return {json.loads(l)['rank']: json.loads(l) for l in p.read_text().splitlines()}


def judge(d):
    js = sorted(d.rglob('judge.txt'))
    assert js, f'no judge.txt under {d}'
    return '\n'.join(p.read_text() for p in js if 'NORMAL' in p.parent.name.upper()), '\n'.join(p.read_text() for p in js)


WPAT = r'projected worst-network wall\s+([\d.]+) s \(mean ([\d.]+) s'
res, ok = dict(dir=str(D), limits=dict(first=LIM[35], warm=LIM[100], worst_wall_s=WALL, setup_s=SETUP)), True
for m in ('normal', 'slow'):
    a = rows(D, m)
    for k, lim in LIM.items():
        r = a[k]['requests']; res[f'{m}_{k}_requests'] = r; passed = r <= lim; ok &= passed
        print(f"{'ok  ' if passed else 'FAIL'} {m:6s} {'first' if k == 35 else 'warm '} requests {r:,} (limit {lim:,}, margin {lim - r:+,})")
normal_text, text = judge(D)
w = [tuple(map(float, x)) for x in re.findall(WPAT, normal_text)]
worst, mean_wall = (max(x[0] for x in w), max(x[1] for x in w)) if w else (None, None)
s0 = [float(x) for x in re.findall(r'projected platform setup wall \(cap 5 s, whole run\)\s+([\d.]+) s', text)]
s1 = [float(x) for x in re.findall(r'projected setup wall under a 1\.3x slow setup\s+([\d.]+) s', text)]
res.update(worst_wall_s=worst, mean_wall_s=mean_wall, setup_s=max(s0) if s0 else None, slow_setup_s=max(s1) if s1 else None)
passed = worst is not None and worst <= WALL; ok &= passed
print(f"{'ok  ' if passed else 'FAIL'} projected worst-network wall {worst} s, mean {mean_wall} s (normal judge; limit {WALL} s)")
for name, v in (('platform setup (whole run)', res['setup_s']), ('setup under a 1.3x slow setup', res['slow_setup_s'])):
    passed = v is not None and v <= SETUP; ok &= passed
    print(f"{'ok  ' if passed else 'FAIL'} projected {name} {v} s (limit {SETUP} s)")
judge_fail = sorted({l.strip() for l in text.splitlines() if l.startswith('  FAIL ')})
ok &= not judge_fail
print(('FAIL judge: ' + judge_fail[0]) if judge_fail else 'ok   official120 judge lines (no FAIL)')
# (a) per-network residual projection vs #336412
b0, b1 = rows(S, 'normal'), rows(D, 'normal')
ratio = {k: (b1[k]['requests'] / b0[k]['requests']) * ((b1[k]['residual_instructions'] / b1[k]['requests']) / (b0[k]['residual_instructions'] / b0[k]['requests'])) for k in (35, 100)}
e = json.load(open(EV))['evaluation']['results']['per_mlp']
tele = [(m_.get('telemetry') or {}) for m_ in e]
proj = sorted(t['residual_wall_time_s'] * ratio[35 if t.get('participant_setup_wall_s') is not None else 100] for t in tele if t.get('residual_wall_time_s'))
ref = sorted(t['residual_wall_time_s'] for t in tele if t.get('residual_wall_time_s'))
med_res = st.median(ref) * ratio[100]
s2_mean = [tuple(map(float, x)) for x in re.findall(WPAT, judge(S)[0])]
wall_scale = mean_wall / max(x[1] for x in s2_mean) if (mean_wall and s2_mean) else None
med_wall_ref = st.median(t['participant_roundtrip_wall_s'] for t in tele if t.get('participant_roundtrip_wall_s'))
med_wall = med_wall_ref * wall_scale if wall_scale else None
res['residual_projection'] = dict(mean=st.mean(proj), median=st.median(proj), p95=proj[int(.95 * len(proj))], max=proj[-1], ratio_first=ratio[35], ratio_warm=ratio[100],
                                  ref_mean=st.mean(ref), ref_median=st.median(ref), ref_p95=ref[int(.95 * len(ref))], ref_max=ref[-1], over_0p4=sum(x > RCAP for x in proj),
                                  median_7p77us_warm=b1[100]['requests'] * US, median_7p77us_first=b1[35]['requests'] * US)
res['wall_projection'] = dict(ref_median_roundtrip_s=med_wall_ref, judge_mean_ratio_vs_s2tr=wall_scale, median_roundtrip_s=med_wall)
print('residual projection vs #336412: median %.3f mean %.3f p95 %.3f max %.3f (ref %.3f / %.3f / %.3f / %.3f); ratios first x%.4f warm x%.4f; '
      'warm x 7.77 us = %.3f s; nets over 0.4 s: %d' % (st.median(proj), st.mean(proj), proj[int(.95 * len(proj))], proj[-1], st.median(ref), st.mean(ref),
      ref[int(.95 * len(ref))], ref[-1], ratio[35], ratio[100], b1[100]['requests'] * US, res['residual_projection']['over_0p4']))
print(f"wall projection: #336412 median round trip {med_wall_ref:.2f} s x judge mean ratio {wall_scale:.4f} = {med_wall:.2f} s")
# (c) noise model: every fired run since 2026-10-13 with >= 50 residual readings; per network i (the same 100 nets every run):
# offset = log(x / run median) = network effect (mean over runs) + noise; noise kept as (residual, wall) pairs, split by the run's started_at
# before / after 2026-10-16T00:00Z. P_i = share of the regime's noise pairs that push net i over 0.4 s residual or 120 s round trip.
runs = []
for f in sorted(glob.glob(str(O/'fire/*/eval_*.json'))):
    ev = json.load(open(f))['evaluation']
    if ev['started_at'] < SINCE:
        continue
    t_ = {m_['mlp_index']: (m_.get('telemetry') or {}) for m_ in (ev.get('results') or {}).get('per_mlp') or []}
    rr = [t for t in t_.values() if t.get('residual_wall_time_s')]
    if len(rr) < 50:
        continue
    mr, mw = st.median(t['residual_wall_time_s'] for t in rr), st.median(t['participant_roundtrip_wall_s'] for t in rr)
    off = {i: (math.log(t['residual_wall_time_s'] / mr) if t.get('residual_wall_time_s') else None,
               math.log(t['participant_roundtrip_wall_s'] / mw) if t.get('participant_roundtrip_wall_s') else None) for i, t in t_.items()}
    runs.append(dict(sub=ev['submission_id'], state=ev['state'], post=ev['started_at'] >= CUT, off=off))
eff = [tuple(st.mean(v) if v else 0.0 for v in ([r_['off'][i][k] for r_ in runs if i in r_['off'] and r_['off'][i][k] is not None] for k in (0, 1))) for i in range(100)]
IN = {'family': lambda r_: r_['sub'] in FAMILY, 'post': lambda r_: r_['post'], 'pre': lambda r_: not r_['post']}
pool = {g: [(r_['off'][i][0] - eff[i][0] if r_['off'][i][0] is not None else None, r_['off'][i][1] - eff[i][1] if r_['off'][i][1] is not None else None)
            for r_ in runs if IN[g](r_) for i in range(100) if i in r_['off']] for g in IN}


def p_fail(mres, mwall, regime):
    tr, tw = math.log(RCAP / mres), (math.log(WCAP / mwall) if mwall else math.inf)
    P = pool[regime]; n = len(P); out = {}
    for name, both in (('residual', (1, 0)), ('wall', (0, 1)), ('either', (1, 1))):
        pi = [sum(1 for e_, u_ in P if (both[0] and e_ is not None and eff[i][0] + e_ > tr) or (both[1] and u_ is not None and eff[i][1] + u_ > tw)) / n for i in range(100)]
        out[name] = dict(any=1 - math.prod(1 - x for x in pi), public=1 - math.prod(1 - x for x in pi[:50]), worst_nets=sorted(range(100), key=lambda i: -pi[i])[:5])
    hit = lambda r_, i: (r_['off'][i][0] is not None and r_['off'][i][0] > tr) or (r_['off'][i][1] is not None and r_['off'][i][1] > tw)
    boot = [(any(hit(r_, i) for i in range(100) if i in r_['off']), any(hit(r_, i) for i in range(50) if i in r_['off'])) for r_ in runs if IN[regime](r_)]
    out['run_boot'] = dict(any=sum(b[0] for b in boot) / len(boot), public=sum(b[1] for b in boot) / len(boot), runs=len(boot))
    return out


res['noise'] = {}
for regime in IN:
    pr = sorted(e_ for e_, u_ in pool[regime] if e_ is not None); pw = sorted(u_ for e_, u_ in pool[regime] if u_ is not None)
    q = lambda v, x: v[min(len(v) - 1, int(x * len(v)))]
    subs = [r_['sub'] + ('' if r_['state'] == 'completed' else '(partial)') for r_ in runs if IN[regime](r_)]
    res['noise'][regime] = dict(runs=subs, nets=len(pool[regime]), residual_sd=st.pstdev(pr), residual_q999=q(pr, .999), wall_sd=st.pstdev(pw), wall_q999=q(pw, .999))
    print(f"noise {regime:4s} (network effects removed): {len(subs)} runs, {len(pool[regime])} nets; residual sd {st.pstdev(pr):.3f} q99.9 {q(pr, .999):+.3f}; "
          f"wall sd {st.pstdev(pw):.3f} q99.9 {q(pw, .999):+.3f}")
res['network_effect_sd'] = dict(residual=st.pstdev(e_[0] for e_ in eff), wall=st.pstdev(e_[1] for e_ in eff))
res['p_fail'] = {}
checks = [('THIS BUILD', med_res, med_wall), ('this build, residual = warm requests x 7.77 us', b1[100]['requests'] * US, med_wall),
          ('check: lead family .32 -> 1.3% (post .355 -> 89/70)', .32, None), ('check: lead family .315 -> 0%', .315, None), ('check: post .355 / lead 89/70', .355, None),
          ('check: lead family wall 96.6 s -> ~0%', .30, 96.6), ('check: lead family wall 100 s -> 2.9%', .30, 100.0)]
for label, mr_, mw_ in checks:
    row = {g: p_fail(mr_, mw_, g) for g in IN}; res['p_fail'][label] = dict(median_residual=mr_, median_wall=mw_, **row)
    print(f"P(fail) [{label}] medians {mr_:.4f} s / {mw_ if mw_ is None else round(mw_, 2)} s: " + '; '.join(
          f"{g}: any {100 * row[g]['either']['any']:.1f}% public {100 * row[g]['either']['public']:.1f}% (residual {100 * row[g]['residual']['any']:.1f}/"
          f"{100 * row[g]['residual']['public']:.1f}, wall {100 * row[g]['wall']['any']:.1f}/{100 * row[g]['wall']['public']:.1f}; run bootstrap "
          f"{100 * row[g]['run_boot']['any']:.0f}/{100 * row[g]['run_boot']['public']:.0f}% of {row[g]['run_boot']['runs']})" for g in IN))
res['count_gates_v3_pass'] = ok; print('COUNT GATES v3', 'PASS' if ok else 'FAIL')
out = D/f'COUNT_GATES_v3_{D.name}.json'
if out.exists():
    out = D/f"COUNT_GATES_v3_{D.name}_{subprocess.check_output(['/bin/date', '-u', '+%Y%m%dT%H%M%SZ'], text=True).strip()}.json"
with out.open('x') as f:
    json.dump(res, f, indent=1)
print('wrote', out)
