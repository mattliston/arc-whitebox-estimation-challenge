"""v2 (trainer, 10-12 16:3xZ; lead 16:4xZ / 16:5xZ: radlab's pre-registered S52 v2, written before any S52 data exists): S52 = make_s53_v3.py's
rules (a) + (b) as written, rule (c) OFF, escal OUT, phase 2 off (the chain's PHASE2_ON file is absent).
Why (radlab, S51 phase 2 on A41, fresh to S50): S50's 8 strength arms re-sized at the S50 point give a joint chi2 of 6.1 on 8 df and every
single-arm LOO >= -0.001%: they are spent; the gain sits in the new arms (arrp -0.046%, t 3.6; dk21n 7-10 -0.031%; drsy 11-14 -0.022%); escal
moved nothing. The approved v1 (13 re-sized strength arms + arrp) would cost about 8 h for an expected 0 +- 0.03%.
  config   start = S51's frozen point if S51 was ACCEPTED (CHECK, RESERVE or SECOND verdict), else S50's; the S48 patch {ss -1, ssb 1, rdc 2.0};
           sizing A43 + A44 7100712-839 (phase 1; the A45 + A46 ranges stay in the config for an explicit phase 2 only); check A49 7101096-127.
           Gradient arms dropped: erest, h20, h6 (default), escal, and every gradient arm whose single-arm LOO in S51's solve was >= 0.
  arms     after S52's 64 gradients on A36: (a) learned-array / head arms by jmix (rad_v6 build, Nf settings); (b) band dial arms by S51's strict
           rule (t^2 > 6 on the 64 gradients, both halves of the same sign at |t| > 1.5; bands 1-6 / 7-10 / 11-14 / all) on the native routes of
           make_s53_v3.py (PER_LAYER / ALL_ONLY / PENDING); kappa excluded; (c) none. keep_v1.json + RULES_v1.json before any arm result.
usage (from O/trainer): python -B cert/RAD/S52/make_s52_v2.py config | arms"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HOME = Path('cert/RAD/S52')
PY = '/Users/matt/anaconda3/bin/python'
RAD = 'cert/RAD/rad_v6.py'
utc = lambda: os.popen('date -u +%FT%TZ').read().strip()
read = lambda p: json.loads(Path(p).read_text())
PER_LAYER = ('iso', 'nfs', 'nkl1', 'nkl2', 'nkl3', 'k4la1', 'kc', 's_carry')      # make_s53_v3.py's routes
ALL_ONLY = ('bscale', 'X1', 'ssb', 'k4', 'nfd', 'n6b', 'n6r', 'n6d')
PENDING = {'k21n': 'dk21n_7_10', 'rsy': 'drsy_11_14'}
ENG = [l for l in Path('../TEAM_LOG.md').read_text().splitlines() if l.startswith('[engineer] ')]
GRAD_ARMS = ('arr', 'arrp', 'b1014', 'b15', 'b19', 'b7es', 'b7f', 'b7k', 'erest', 'escal', 'h17', 'h20', 'h6', 'hb3')
mode = sys.argv[1]


def route(fam):
    if fam in PER_LAYER:
        return 'layer'
    if fam in ALL_ONLY:
        return 'all'
    if fam in PENDING:
        if any('S51 PORT PARITY %s: FAILED' % PENDING[fam] in l for l in ENG):
            return None
        return 'layer' if any('S51 PORT PARITY %s: PASS' % PENDING[fam] in l for l in ENG) else 'all'
    return None


def accepted(rnd):
    files = [Path('cert/RAD/%s/%s' % (rnd, f)) for f in ('CHECK_VERDICT_v1.json', 'RESERVE_VERDICT_v1.json', 'SECOND_VERDICT_v1.json')]
    return any(p.exists() and read(p).get('accepted') is True for p in files)


if mode == 'config':
    out = HOME / 'config.json'
    assert not out.exists()
    src = 'S51' if accepted('S51') else 'S50'
    f = read('cert/RAD/%s/FROZEN_POINT_v1.json' % src)
    last = {}
    p = Path('cert/RAD/S51/solve_v1.json')
    if p.exists():
        last = {a: v for a, v in read(p)['arm_alone_loo_pct'].items() if a in GRAD_ARMS}
    drop = ['erest', 'h20', 'h6', 'escal'] + sorted(a for a, v in last.items() if v >= 0 and a not in ('erest', 'h20', 'h6', 'escal'))
    cfg = dict(name='S52', note='trainer RAD S52 v2 (lead 16:5xZ, radlab): start = %s frozen point; rules (a) + (b) of make_s53_v3, (c) off, escal out; '
                                'sizing A43 + A44 (phase 2 only with the chain\'s PHASE2_ON file), check A49 1096-1127.' % src,
               patch={'ss': -1, 'ssb': 1, 'rdc': 2.0}, params=f['params'], hu=f['head'], bill_saving_pct=0.0, drop_arms=','.join(drop),
               sizing=[[7100712, 7100775], [7100776, 7100839], [7100840, 7100903], [7100904, 7100967]], sizing_source='fresh',
               check_seeds=[7101096, 7101127], decide='ab', check_nd=False)
    out.write_text(json.dumps(cfg, indent=1))
    print('S52 config (v2): start = %s frozen point (%s); gradient arms dropped: %s' % (src, f['params'], ','.join(drop)))
elif mode == 'arms':
    import torch
    cfg = read(HOME / 'config.json')
    seeds = [int(x) for x in Path('cert/Nf/gen_allow_grad_v1.txt').read_text().split(',')]
    rows = [torch.load('run_RAD_S52_gen/results/a_RS52g_r%d.pt' % r, map_location='cpu', weights_only=False) for r in seeds]
    assert len(rows) == 64
    half = 32
    t = lambda x: float(x.mean() / (x.std(ddof=1) / len(x) ** .5)) if x.std() > 0 else 0.0
    routes = {fam: route(fam) for fam in rows[0]['pl_grads']}
    dials = []
    for fam in sorted(rows[0]['pl_grads'], key=str.lower):      # (b) band dial arms, strict rule, native routes only
        if routes[fam] is None or fam == 'kappa':
            continue
        G = np.stack([r['pl_grads'][fam].double().numpy() * (1e8 / r['scale']) for r in rows])
        if not np.abs(G).max() > 0:
            continue
        for band, (lo, hi) in (('1-6', (1, 6)), ('7-10', (7, 10)), ('11-14', (11, 14)), ('all', (0, 15))):
            if routes[fam] == 'all' and band != 'all':
                continue
            s = G[:, lo:hi + 1].sum(1); ta, tb, tt = t(s[:half]), t(s[half:]), t(s)
            if tt * tt > 6 and ta * tb > 0 and min(abs(ta), abs(tb)) > 1.5:
                dials.append(fam if band == 'all' else '%s:%d:%d' % (fam, lo, hi))
    DG = HOME / 'arms_grad_v1'
    subprocess.run([PY, '-B', RAD, 'build', str(HOME / 'config.json'), str(DG), ','.join(dials) or '-', '-'], check=True)   # (a) + (b)
    keep = read(DG / 'arms.json')['live']
    stamp = utc()
    (HOME / 'keep_v1.json').write_text(json.dumps(dict(keep=keep, utc=stamp, dirs=[str(DG)]), indent=1))
    (HOME / 'RULES_v1.json').write_text(json.dumps(dict(
        schema='trainer_RAD_rules_v1', utc=stamp, round='S52', version='make_s52_v2.py (lead 16:5xZ, radlab)', written_before_any_arm_result=True,
        start=cfg['params'], patch=cfg['patch'],
        sizing='phase 1 = A43 7100712-775 + A44 7100776-839 (128 fresh; truth-noise-corrected %); phase 2 (A45 + A46) only with the chain\'s PHASE2_ON file',
        gradients='64 at the start point on A36 7100264-327', arms=keep, gradient_arms_dropped=cfg['drop_arms'],
        dial_rule='t^2 > 6 on 64 gradients with both halves of the same sign at |t| > 1.5; on the native routes only (make_s53_v3.py)',
        routes={f: r for f, r in sorted(routes.items()) if r}, dials=dials, strength_rule='OFF (lead 16:5xZ: S50\'s strength arms are spent)',
        port_rule='cert/RAD/S52/PORT_RULE_v1.json',
        solve='ridge + leave-one-network-out (rad_v6.py freeze; candidates all / no dials / no fixed / fixed only); null if LOO >= 0',
        check='A49 7101096-127 (16M): frozen vs the unrefitted start, ACCEPT iff pooled change < -SE; sealed A49 7101128-159 = the physicist'), indent=1))
    print('S52 arms v2 (%d): %s | dials %s | routes %s' % (len(keep), ' '.join(keep), dials, {f: r for f, r in sorted(routes.items()) if r}))
