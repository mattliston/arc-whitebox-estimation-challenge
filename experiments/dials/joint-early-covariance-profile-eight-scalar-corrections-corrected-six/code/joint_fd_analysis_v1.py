"""dissect337259 agent 2026-10-20: analysis of the lead's joint FD run (IDEAS_v3 section 3). Reads eval_joint_fd_*.log (lines
'  seed N {json} U u bill b'), maps each line's json to its index in joint_fd_settings_v1.json (exact dict equality) and its label in
joint_fd_labels_v1.json. Per net: ln U per label; dial slopes a = (lnU+ - lnU-)/2 and curvatures b = lnU+ + lnU- - 2 lnU(base) at B0 and B1;
pooled mean / se / sign counts; regime test (B1 - B0); substitute / complement classification; PAIR cross terms; diagonal Newton step at B1
with the block rule and damping; the joint setting dict for the held-out run. usage: joint_fd_analysis_v1.py [--selftest] [logs...]"""
import json, re, sys, glob, copy
import numpy as np
K = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/'
settings = json.load(open(K + 'joint_fd_settings_v1.json')); labels = json.load(open(K + 'joint_fd_labels_v1.json'))
assert len(settings) == len(labels)
canon = [json.dumps(s, sort_keys=True) for s in settings]
args = [a for a in sys.argv[1:] if not a.startswith('--')]; selftest = '--selftest' in sys.argv
logs = args if args else sorted(glob.glob(K + 'eval_joint_fd_?.log'))   # A/B/C only: the *_crash2_strkeys.log files of the invalid 17:12Z run must not be read
U = {}   # seed -> {label: U}
if selftest:
    rng = np.random.default_rng(0)
    for seed in (1, 2, 3, 4, 5, 6):
        base = 1.3e-8 * (1 + 0.1 * rng.standard_normal()); U[seed] = {}
        for lab in labels:
            eff = -0.003 if lab.startswith('B1') else (-0.002 if lab.startswith('B2') else 0.0)
            eff += (-0.002 if ('gsk_early +' in lab or 'kc_early +' in lab) else 0.0) + (0.0015 if 'edge_late +' in lab and lab.startswith('B1') else 0.0)
            U[seed][lab] = base * (1 + eff + 0.001 * rng.standard_normal())
else:
    for f in logs:
        for ln in open(f):
            m = re.match(r'\s+seed (\d+) (\{.*\}) U ([0-9.e+-]+) bill', ln)
            if not m: continue
            try: idx = canon.index(json.dumps(json.loads(m.group(2)), sort_keys=True))
            except ValueError: continue
            U.setdefault(int(m.group(1)), {})[labels[idx]] = float(m.group(3))
seeds = sorted(s for s in U if 'B0' in U[s] and 'B1' in U[s]); n = len(seeds)
print('nets with B0 and B1:', seeds)
def lnU(seed, lab): return np.log(U[seed][lab]) if lab in U[seed] else np.nan
dials = ['gsk_early', 'kc_early', 'iso_early', 'k21n_early', 'gk4_early', 'edge_late', 'iso_late', 'k31']
# 1. regime
d10 = np.array([100 * (lnU(s, 'B1') - lnU(s, 'B0')) for s in seeds])
print('REGIME: B1 - B0 = %+.3f%% (se %.3f, lower %d/%d): %s' % (d10.mean(), d10.std(ddof=1) / np.sqrt(n), (d10 < 0).sum(), n, 'S (structural)' if d10.mean() < -0.2 and (d10 < 0).sum() >= max(1, int(np.ceil(0.66 * n))) else 'C (co-adapted) or undecided'))
# 1b. profile magnitude from the three points c = 0 (B0), h (B1), 2h (B2): f = U0 + a c + b c^2 -> c* = h (3U0 - 4Uh + Uf) / (2 (U0 - 2Uh + Uf))
b2lab = next((x for x in labels if x.startswith('B2')), None)
if b2lab is not None and all(b2lab in U[s] for s in seeds):
    cs = []; gs = []
    for s in seeds:
        U0, Uh, Uf = lnU(s, 'B0'), lnU(s, 'B1'), lnU(s, b2lab)
        den = U0 - 2 * Uh + Uf
        if den > 0:
            cst = 0.5 * (3 * U0 - 4 * Uh + Uf) / (2 * den); cst = float(np.clip(cst, -1.0, 2.0))   # in units of the FULL profile (h = 0.5)
            a = (-3 * U0 + 4 * Uh - Uf) / (2 * 0.5); b = den / (2 * 0.25); gs.append(100 * (a * cst + b * cst * cst)); cs.append(cst)
        else:
            cs.append(float(np.nan)); gs.append(float(np.nan))
    cs = np.array(cs); gs = np.array(gs)
    d20 = np.array([100 * (lnU(s, b2lab) - lnU(s, 'B0')) for s in seeds])
    print('PROFILE MAGNITUDE: B2 - B0 = %+.3f%% (lower %d/%d); per-net c* (units of the full profile, convex nets only): %s; pooled parabola c* = %.2f, gain at c* %+.3f%%' % (
        d20.mean(), (d20 < 0).sum(), n, ', '.join('%.2f' % c if np.isfinite(c) else 'concave' for c in cs),
        (lambda U0, Uh, Uf: 0.5 * (3 * U0 - 4 * Uh + Uf) / (2 * (U0 - 2 * Uh + Uf)) if (U0 - 2 * Uh + Uf) > 0 else float('nan'))(np.mean([lnU(s, 'B0') for s in seeds]), np.mean([lnU(s, 'B1') for s in seeds]), np.mean([lnU(s, b2lab) for s in seeds])),
        np.nanmean(gs) if np.isfinite(gs).any() else float('nan')))
# 2. slopes and curvatures
res = {}
print('%-11s | %8s %6s %5s | %8s %6s %5s | %8s | %8s %8s | %s' % ('dial', 'a(B0)%', 'se', 'sgn', 'a(B1)%', 'se', 'sgn', 'D=a1-a0', 'b(B0)', 'b(B1)', 'class'))
for dname in dials:
    rows = {}
    for B in ('B0', 'B1'):
        a = []; b = []
        for s in seeds:
            lp, lm, l0 = lnU(s, '%s %s +' % (B, dname)), lnU(s, '%s %s -' % (B, dname)), lnU(s, B)
            if np.isfinite(lp) and np.isfinite(lm) and np.isfinite(l0): a.append(100 * (lp - lm) / 2); b.append(100 * (lp + lm - 2 * l0))
        rows[B] = (np.array(a), np.array(b))
    a0, b0 = rows['B0']; a1, b1 = rows['B1']
    if len(a0) < 2 or len(a1) < 2: print('%-11s incomplete' % dname); continue
    se0, se1 = a0.std(ddof=1) / np.sqrt(len(a0)), a1.std(ddof=1) / np.sqrt(len(a1))
    real0 = abs(a0.mean()) > 2.5 * se0; real1 = abs(a1.mean()) > 2.5 * se1
    D = a1.mean() - a0.mean()
    if real0 and abs(a1.mean()) < abs(a0.mean()) and np.sign(a1.mean()) == np.sign(a0.mean()) or (real0 and not real1): cls = 'SUBSTITUTE (slope relaxes at B1)'
    elif real1 and not real0: cls = 'COMPLEMENT (slope appears at B1)'
    elif real0 and real1 and abs(a1.mean()) > abs(a0.mean()): cls = 'COMPLEMENT (slope grows at B1)'
    elif not real0 and not real1: cls = 'flat'
    else: cls = 'mixed'
    res[dname] = dict(a0=a0.mean(), a1=a1.mean(), se0=se0, se1=se1, b0=b0.mean(), b1=b1.mean(), real1=real1, cls=cls)
    print('%-11s | %+8.3f %6.3f %2d/%-2d | %+8.3f %6.3f %2d/%-2d | %+8.3f | %+8.3f %+8.3f | %s' % (dname, a0.mean(), se0, (a0 < 0).sum(), len(a0), a1.mean(), se1, (a1 < 0).sum(), len(a1), D, b0.mean(), b1.mean(), cls))
# 3. pair cross terms at B1: cross = lnU(pair) - lnU(B1) - [lnU(i) - lnU(B1)] - [lnU(j) - lnU(B1)]
print('PAIR cross terms at B1 (x100 ln; positive = the two moves interfere):')
for lab in labels:
    if not lab.startswith('B1 PAIR'): continue
    parts = lab[len('B1 PAIR '):].split(' & '); cr = []
    for s in seeds:
        li = lnU(s, 'B1 ' + parts[0]); lj = lnU(s, 'B1 ' + parts[1]); lp = lnU(s, lab); l1 = lnU(s, 'B1')
        if all(np.isfinite(x) for x in (li, lj, lp, l1)): cr.append(100 * ((lp - l1) - (li - l1) - (lj - l1)))
    if cr: cr = np.array(cr); print('  %-36s pair %+.3f%%  sum of singles %+.3f%%  cross %+.3f (se %.3f)' % (lab, np.mean([100 * (lnU(s, lab) - lnU(s, 'B1')) for s in seeds]), np.mean([100 * ((lnU(s, 'B1 ' + parts[0]) - lnU(s, 'B1')) + (lnU(s, 'B1 ' + parts[1]) - lnU(s, 'B1'))) for s in seeds]), cr.mean(), cr.std(ddof=1) / np.sqrt(len(cr))))
# 4. diagonal Newton at B1 (units of the FD step; clipped to +-1), block rule, damping 0.5
bmin = np.median([r['b1'] for r in res.values() if r['b1'] > 0]) if any(r['b1'] > 0 for r in res.values()) else 0.05
steps = {}
for dname, r in res.items():
    if not r['real1']: continue
    bb = max(r['b1'], bmin); t = float(np.clip(-r['a1'] / bb, -1, 1)); g = r['a1'] * t + 0.5 * bb * t * t
    steps[dname] = (t, g)
early = [d for d in steps if d.endswith('_early')]; late = [d for d in steps if d not in early]
print('Newton steps at B1 (t in FD-step units, predicted gain %%): ' + ', '.join('%s t=%+.2f g=%+.3f' % (d, t, g) for d, (t, g) in steps.items()))
chosen = {}
for block in (early, late):
    if not block: continue
    k = len(block); scale = 1 / np.sqrt(k)
    for d in block: chosen[d] = 0.5 * scale * steps[d][0]     # damping 0.5 x block rule
pred = sum(res[d]['a1'] * chosen[d] + 0.5 * max(res[d]['b1'], bmin) * chosen[d] ** 2 for d in chosen)
print('JOINT STEP (damped, block-scaled), FD-step units: %s ; predicted gain vs B1 %+.3f%% ; vs B0 %+.3f%%' % ({d: round(t, 2) for d, t in chosen.items()}, pred, pred + d10.mean()))
# 5. the held-out setting: start from B1 and move each chosen dial by t x (its + setting minus B1 value)
def move(base, plus, t):
    out = copy.deepcopy(base)
    for key in plus:
        if isinstance(plus[key], dict) and isinstance(base.get(key), dict):
            for kk, v in plus[key].items():
                bv = base[key].get(kk, None)
                if isinstance(v, (int, float)) and isinstance(bv, (int, float)): out[key][kk] = bv + t * (v - bv)
                elif isinstance(v, (int, float)) and bv is None and key in ('_GSK_BY_LAYER', '_GK4_BY_LAYER'): out[key][kk] = 1.0 + t * (v - 1.0)
        elif isinstance(plus[key], (int, float)) and isinstance(base.get(key), (int, float)): out[key] = base[key] + t * (plus[key] - base[key])
    return out
B1 = settings[labels.index('B1')]; joint = copy.deepcopy(B1)
for d, t in chosen.items():
    plus = settings[labels.index('B1 %s +' % d)]; joint = move(joint, plus, t)
json.dump([B1, joint], open(K + 'agent_ideas_v1/joint_step_settings_v1.json', 'w'))
print('held-out settings written: agent_ideas_v1/joint_step_settings_v1.json  ([B1, B1 + joint step]; run with the base as setting 0 if a B0 reference is wanted)')
