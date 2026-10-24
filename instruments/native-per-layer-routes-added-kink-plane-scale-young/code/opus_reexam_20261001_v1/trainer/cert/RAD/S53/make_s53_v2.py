"""v2 (10-12 07:5xZ, lead 07:5xZ + the engineer's route list of 07:4xZ, written before any S53 data): the dial families of rule (b) follow the
NATIVE ROUTES exactly. PER_LAYER (bands 1-6 / 7-10 / 11-14 / all): iso, nfs, nkl1, nkl2, nkl3, k4la1, kc (certified per layer, carried per layer
by the exporter); k21n / rsy per layer only if the engineer posted 'S51 PORT PARITY dk21n_7_10: PASS' / '... drsy_11_14: PASS', none of
them if he posted FAILED, the 'all' band only otherwise. ALL_ONLY (one global shift = one literal): bscale, X1 (per-layer route built, not
certified), s_carry (the exporter writes it as the dial literal), ssb (per layer never exercised), k4 (native bands 1-7 / 8-11 / 12-14 differ
from ours), nfd, n6b, n6r, n6d (single literals natively). Every other family is out of (b) (rso: no export form; kappa; edx: bill).
Rule (c) also leaves out any S52 arm the engineer marked 'S52 PORT PARITY <arm>: FAILED'. The export is export_dials_s46_v5.py.
v1: TAG opus-trainer RAD S53 (lead 10-12 07:4xZ: APPROVED as proposed at 07:3xZ; the template of S51): the pre-declared, rule-driven construction
of round S53, run by cert/RAD/S53/chain_s53.sh.
  config   start = the latest ACCEPTED frozen point of S52 / S51 / S50 (accepted = that round's CHECK, RESERVE or SECOND verdict file says so);
           the S48 patch {ss -1, ssb 1, rdc 2.0}; sizing on FRESH 8M blocks (truth-noise-corrected % via rad_v6.py): phase 1 = A48 7101032-095
           + A50 7101160-223, phase 2 = A51 7101224-287 + A53 7101352-415 (the chain's 12 h rule); check on A52's first half 7101288-319 (16M)
           vs the unrefitted start, ACCEPT iff pooled change < -SE; sealed = A52's second half 7101320-351 (the physicist). A47 7100968-1031
           stays an unassigned reserve. Gradients for the learned-array and dial arms: 64 at the start on A36 7100264-327, as every round.
  arms     after the gradients, by fixed rules:
           (a) learned-array / head arms: cert/refit/jmix_generated_v7.py via rad_v6.py build with the Nf settings; dropped: erest, h20, h6
               (the default) and every gradient arm whose MOST RECENT single-arm LOO was >= 0 (S52's solve, else S51's);
           (b) band dial arms by S51's strict rule (t^2 > 6 on the 64 gradients with both halves of the same sign at |t| > 1.5; bands 1-6 /
               7-10 / 11-14 / all) on the families and bands with a native route (v2: PER_LAYER / ALL_ONLY / PENDING below); kappa excluded;
           (c) strength arms: S52's forward arms with single-arm LOO < 0 on S52's sizing networks (cert/RAD/S52/solve_v1.json), rebuilt from
               the start with S52's trial shifts (S51's forward arms by the same rule if S52 has no solve); kappa excluded.
           writes keep_v1.json and RULES_v1.json before any arm result exists. The port rule is cert/RAD/S53/PORT_RULE_v1.json.
usage (from O/trainer): python -B cert/RAD/S53/make_s53.py config | arms"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HOME = Path('cert/RAD/S53')
PY = '/Users/matt/anaconda3/bin/python'
RAD = 'cert/RAD/rad_v6.py'
utc = lambda: os.popen('date -u +%FT%TZ').read().strip()
read = lambda p: json.loads(Path(p).read_text())
PER_LAYER = ('iso', 'nfs', 'nkl1', 'nkl2', 'nkl3', 'k4la1', 'kc')      # v2: engineer 07:4xZ, certified per layer; the exporter carries them per layer
ALL_ONLY = ('bscale', 'X1', 's_carry', 'ssb', 'k4', 'nfd', 'n6b', 'n6r', 'n6d')      # v2: one global shift only
PENDING = {'k21n': 'dk21n_7_10', 'rsy': 'drsy_11_14'}      # v2: per layer iff the engineer's S51 parity line says PASS; out if FAILED; else 'all'
ENG = [l for l in Path('../TEAM_LOG.md').read_text().splitlines() if l.startswith('[engineer] ')]


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


GRAD_ARMS = ('arr', 'arrp', 'b1014', 'b15', 'b19', 'b7es', 'b7f', 'b7k', 'erest', 'escal', 'h17', 'h20', 'h6', 'hb3')
SIZING = [[7101032, 7101095], [7101160, 7101223], [7101224, 7101287], [7101352, 7101415]]
mode = sys.argv[1]


def accepted(rnd):
    files = [Path('cert/RAD/%s/%s' % (rnd, f)) for f in ('CHECK_VERDICT_v1.json', 'RESERVE_VERDICT_v1.json', 'SECOND_VERDICT_v1.json')]
    return any(p.exists() and read(p).get('accepted') is True for p in files)


if mode == 'config':
    out = HOME / 'config.json'
    assert not out.exists()
    src = next(r for r in ('S52', 'S51', 'S50') if accepted(r))
    f = read('cert/RAD/%s/FROZEN_POINT_v1.json' % src)
    last = {}
    for rnd in ('S51', 'S52'):      # the most recent single-arm LOO of each gradient arm: S52's overrides S51's
        p = Path('cert/RAD/%s/solve_v1.json' % rnd)
        if p.exists():
            last.update({a: v for a, v in read(p)['arm_alone_loo_pct'].items() if a in GRAD_ARMS})
    drop = ['erest', 'h20', 'h6'] + sorted(a for a, v in last.items() if v >= 0 and a not in ('erest', 'h20', 'h6'))
    cfg = dict(name='S53', note='trainer RAD S53 (lead 10-12 07:4xZ): start = %s frozen point; arms by the rules of make_s53.py; sizing A48 + A50 '
                                '(+ A51 + A53 under the 12 h rule), check A52 7101288-319.' % src,
               patch={'ss': -1, 'ssb': 1, 'rdc': 2.0}, params=f['params'], hu=f['head'], bill_saving_pct=0.0, drop_arms=','.join(drop),
               sizing=SIZING, sizing_source='fresh', check_seeds=[7101288, 7101319], decide='ab', check_nd=False)
    out.write_text(json.dumps(cfg, indent=1))
    print('S53 config: start = %s frozen point (%s); gradient arms dropped: %s' % (src, f['params'], ','.join(drop)))
elif mode == 'arms':
    import torch
    cfg = read(HOME / 'config.json')
    seeds = [int(x) for x in Path('cert/Nf/gen_allow_grad_v1.txt').read_text().split(',')]
    rows = [torch.load('run_RAD_S53_gen/results/a_RS53g_r%d.pt' % r, map_location='cpu', weights_only=False) for r in seeds]
    assert len(rows) == 64
    half = 32
    t = lambda x: float(x.mean() / (x.std(ddof=1) / len(x) ** .5)) if x.std() > 0 else 0.0
    dials = []
    routes = {fam: route(fam) for fam in rows[0]['pl_grads']}
    for fam in sorted(rows[0]['pl_grads'], key=str.lower):      # (b) band dial arms, strict rule, on the native routes only (v2)
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
    src = 'S52' if Path('cert/RAD/S52/solve_v1.json').exists() else 'S51'      # (c) strength arms with single-arm LOO < 0
    solve, sizes = read('cert/RAD/%s/solve_v1.json' % src), read('cert/RAD/%s/arms_fixed_v1/arms.json' % src)['size']
    failed = [a for a in sizes if any('%s PORT PARITY %s: FAILED' % (src, a) in l for l in ENG)]      # v2: arms whose port failed
    strength = [a for a, v in solve['arm_alone_loo_pct'].items() if v < 0 and a in sizes and not a.startswith('fkappa') and a not in failed]
    spec = []
    for a in strength:
        fam, lo, hi = a[1:].rsplit('_', 2)
        spec.append('%s:%s:%s:%s' % (fam, lo, hi, sizes[a]))
    DF = HOME / 'arms_fixed_v1'
    if spec:
        subprocess.run([PY, '-B', RAD, 'fixed', str(HOME / 'config.json'), str(DF), ','.join(spec)], check=True)
    keep = read(DG / 'arms.json')['live'] + (read(DF / 'arms.json')['live'] if spec else [])
    stamp = utc()
    (HOME / 'keep_v1.json').write_text(json.dumps(dict(keep=keep, utc=stamp, dirs=[str(DG)] + ([str(DF)] if spec else [])), indent=1))
    (HOME / 'RULES_v1.json').write_text(json.dumps(dict(
        schema='trainer_RAD_rules_v1', utc=stamp, round='S53', written_before_any_arm_result=True, start=cfg['params'], patch=cfg['patch'],
        sizing='phase 1 = A48 7101032-095 + A50 7101160-223 (128 fresh, 8M: truth-noise-corrected %); phase 2 = A51 7101224-287 + A53 7101352-415 '
               'only if the freeze projected from phase 1 lands within 12 h of the round start (chain_s53.sh), else freeze on phase 1',
        gradients='64 at the start point on A36 7100264-327', arms=keep, gradient_arms_dropped=cfg['drop_arms'],
        dial_rule='t^2 > 6 on 64 gradients with both halves of the same sign at |t| > 1.5; on the native routes only (make_s53_v2.py)',
        routes={f: r for f, r in sorted(routes.items()) if r}, dials=dials, strength_port_failed=failed, strength_rule="%s's forward arms with single-arm LOO < 0 on %s's sizing networks" % (src, src),
        strength=strength, port_rule='cert/RAD/S53/PORT_RULE_v1.json',
        solve='ridge + leave-one-network-out (rad_v6.py freeze; candidates all / no dials / no fixed / fixed only); null if LOO >= 0',
        check='A52 7101288-319 (16M): frozen vs the unrefitted start, ACCEPT iff pooled change < -SE; sealed A52 7101320-351 = the physicist'), indent=1))
    print('S53 arms (%d): %s | dials %s | routes %s | strength (%s) %s' % (len(keep), ' '.join(keep), dials, {f: r for f, r in sorted(routes.items()) if r}, src, strength))
