"""TAG opus-trainer RAD S51 (lead 10-12 ~01:xxZ): the pre-declared, rule-driven construction of round S51, run by cert/RAD/S51/chain.sh.
  config   start point by S50's verdict (ACCEPTED -> S50's frozen point; otherwise the S48 card = the MS point), the S48 patch {ss -1, ssb 1, rdc 2.0},
           sizing on A39 7100456-519 + A41 7100584-647 (128 fresh no-plan networks), check on A42 7100648-679 vs the unrefitted start (sealed A42 680-711
           = the physicist's stage 2). Gradients for the learned-array arms: 64 at the start point on A36 7100264-327 (no sizing network among them).
  arms     after the gradients, by fixed rules:
           (a) learned-array / head arms: cert/refit/jmix_generated_v7.py via rad.py build with the Nf settings (arrays split, edge scalars, capped trials);
           (b) band dial arms: every per-layer family and band (1-6 / 7-10 / 11-14 / all) whose summed gradient has t^2 > 6 with both split halves of the
               same sign at |t| > 1.5 (the strict rule of cert/RAD/band_table.py);
           (c) strength arms: the forward arms of S50 whose single-arm LOO on S50's 64 sizing networks was < 0 (cert/RAD/S50/solve_v1.json), rebuilt
               from the start point with S50's trial shifts.
           writes keep_v1.json (all arms built) and RULES_v1.json before any arm result exists.
v2 (10-12 04:35Z, lead): S50's acceptance = the POOLED reserve verdict (cert/RAD/S50/RESERVE_VERDICT_v1.json: A40 + A38, 64 networks) when it exists;
the 'kappa' family is excluded everywhere (its native route failed parity: the strength arm fkappa_1_14 and any kappa band dial).
usage (from O/trainer): python -B cert/RAD/S51/make_s51.py config | arms"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HOME = Path('cert/RAD/S51')
PY = '/Users/matt/anaconda3/bin/python'
utc = lambda: os.popen('date -u +%FT%TZ').read().strip()
read = lambda p: json.loads(Path(p).read_text())
mode = sys.argv[1]

if mode == 'config':
    out = HOME / 'config.json'
    assert not out.exists()
    nd = read('cert/Nd/FROZEN_POINT_v1.json')
    v50, r50 = Path('cert/RAD/S50/CHECK_VERDICT_v1.json'), Path('cert/RAD/S50/RESERVE_VERDICT_v1.json')
    accepted = read(r50)['accepted'] if r50.exists() else (v50.exists() and read(v50)['accepted'])      # v2: the pooled reserve verdict decides
    if accepted:
        f = read('cert/RAD/S50/FROZEN_POINT_v1.json'); params, hu, start = f['params'], f['head'], 'S50 frozen point (ACCEPTED)'
    else:
        params, hu, start = 'cert/RAD/MS/point_v1/params.pt', nd['head'], 'the S48 card = MS point (S50 not accepted)'
    cfg = dict(name='S51', note='trainer RAD S51 (lead 10-12): start = %s; learned-array + band-dial + strength arms by the rules of make_s51.py; '
                                'sized on A39 + A41 (128), checked on A42 7100648-679 vs the unrefitted start.' % start,
               patch={'ss': -1, 'ssb': 1, 'rdc': 2.0}, params=params, hu=hu, bill_saving_pct=0.0,
               sizing=[[7100456, 7100519], [7100584, 7100647]], sizing_source='fresh', check_seeds=[7100648, 7100679], decide='ab', check_nd=False)
    out.write_text(json.dumps(cfg, indent=1))
    print('S51 config: start =', start)
elif mode == 'arms':
    import torch
    cfg = read(HOME / 'config.json')
    seeds = [int(x) for x in Path('cert/Nf/gen_allow_grad_v1.txt').read_text().split(',')]
    rows = [torch.load('run_RAD_S51_gen/results/a_RS51g_r%d.pt' % r, map_location='cpu', weights_only=False) for r in seeds]
    assert len(rows) == 64
    N, half = 64, 32
    t = lambda x: float(x.mean() / (x.std(ddof=1) / len(x) ** .5)) if x.std() > 0 else 0.0
    dials = []
    for fam in sorted(rows[0]['pl_grads'], key=str.lower):      # (b) band dial arms, strict rule
        if fam == 'kappa':      # v2: kappa excluded (native route failed parity)
            continue
        G = np.stack([r['pl_grads'][fam].double().numpy() * (1e8 / r['scale']) for r in rows])
        if not np.abs(G).max() > 0:
            continue
        for band, (lo, hi) in (('1-6', (1, 6)), ('7-10', (7, 10)), ('11-14', (11, 14)), ('all', (0, 15))):
            s = G[:, lo:hi + 1].sum(1); ta, tb, tt = t(s[:half]), t(s[half:]), t(s)
            if tt * tt > 6 and ta * tb > 0 and min(abs(ta), abs(tb)) > 1.5:
                dials.append(fam if band == 'all' else '%s:%d:%d' % (fam, lo, hi))
    DG = HOME / 'arms_grad_v1'
    subprocess.run([PY, '-B', 'cert/RAD/rad.py', 'build', str(HOME / 'config.json'), str(DG), ','.join(dials) or '-', '-'], check=True)   # (a) + (b)
    s50 = read('cert/RAD/S50/solve_v1.json')      # (c) strength arms with single-arm LOO < 0 on S50's 64
    sizes = {}
    for d in ('cert/RAD/S48/arms_fixed_v1', 'cert/RAD/S48/arms_fixed_v2', 'cert/RAD/S48/arms_fixed_v3'):
        sizes.update(read(Path(d) / 'arms.json')['size'])
    strength = [a for a, v in s50['arm_alone_loo_pct'].items() if v < 0 and not a.startswith('fkappa')]      # v2: kappa excluded
    spec = []
    for a in strength:
        fam, lo, hi = a[1:].rsplit('_', 2)
        spec.append('%s:%s:%s:%s' % (fam, lo, hi, sizes[a]))
    DF = HOME / 'arms_fixed_v1'
    if spec:
        subprocess.run([PY, '-B', 'cert/RAD/rad.py', 'fixed', str(HOME / 'config.json'), str(DF), ','.join(spec)], check=True)
    keep = read(DG / 'arms.json')['live'] + (read(DF / 'arms.json')['live'] if spec else [])
    stamp = utc()
    (HOME / 'keep_v1.json').write_text(json.dumps(dict(keep=keep, utc=stamp, dirs=[str(DG)] + ([str(DF)] if spec else [])), indent=1))
    (HOME / 'RULES_v1.json').write_text(json.dumps(dict(
        schema='trainer_RAD_rules_v1', utc=stamp, round='S51', written_before_any_arm_result=True, start=cfg['params'], patch=cfg['patch'],
        sizing='A39 7100456-519 + A41 7100584-647 (128 fresh no-plan networks)', gradients='64 at the start point on A36 7100264-327',
        arms=keep, dial_rule='t^2 > 6 on 64 gradients with both halves of the same sign at |t| > 1.5', dials=dials,
        strength_rule="S50's forward arms with single-arm LOO < 0 on S50's 64", strength=strength,
        solve='ridge + leave-one-network-out on the 128 (rad.py freeze; candidates all / no dials / no fixed / fixed only); null if LOO >= 0',
        check='fresh A42 7100648-679: frozen vs the unrefitted start, ACCEPT iff pooled change < -SE; stage 2 = the physicist on sealed A42 680-711'), indent=1))
    print('S51 arms (%d): %s | dials %s | strength %s' % (len(keep), ' '.join(keep), dials, strength))
