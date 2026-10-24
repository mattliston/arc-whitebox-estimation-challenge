"""make_s53_v11.py (written 2026-10-12T18:44:37Z, lead 18:3xZ): = make_s53_v10.py + the ABSORB hook (cert/RAD/absorb_v1.py; inert while cert/RAD/ABSORB.json is absent) + the fagg fix of the strength-arm name parse (cert/RAD/fix_make_absorb_v1.py). Nothing else changed.
make_s53_v10.py (written 2026-10-12T18:34:18Z, lead 18:1xZ): = make_s53_v9.py with the ZERO-ARM GUARD (cert/RAD/zero_arms_v1.py, cert/RAD/fix_make_zero_v1.py): an arm whose parameter file equals the start point bit for bit leaves arms.json 'live' before keep_v1.json and before any arm is queued; one TEAM_LOG line when any is dropped. Nothing else changed.
make_s53_v9.py (written 2026-10-12T17:13:25Z, radlab): = make_s53_v8.py with radlab/ptload.py (imported by the aggregate builder) pinned by sha256 as well.
make_s53_v8.py (written 2026-10-12T17:07:52Z, radlab 17:0xZ): = make_s53_v7.py with rule (d) taken from the frozen radlab aggregate builder (cert/RAD/make_agg_arm_v2.py).
make_s53_v7.py (written 2026-10-12T17:01:11Z, lead 17:0xZ / radlab): = make_s53_v6.py with rule (d), ONE aggregate weak-band arm fagg (cert/RAD/make_agg_arm_v1.py: per family
the largest-|t| routed band with t^2 >= 4 that rule (b) did not take and that overlaps none of its strict bands, moved against its gradient sign by
0.02 x min(1, |t|/6), all in one arm).
make_s53_v6.py (written 2026-10-12T16:51:12Z, lead / radlab 16:5xZ): = make_s53_v5.py with rule (b)'s routes read from cert/RAD/ROUTES.json at the arm stage (adds
per-family band lists: k4 1-7 / 8-11 / 12-14 / 15-15 / all); the strict rule unchanged.
make_s53_v5.py (written 2026-10-12T16:44:26Z, lead 16:5xZ / radlab): = make_s53_v4.py with the start point = the latest round that CONTINUES (cert/RAD/CONTINUE_RULE:
absent or se = accepted by the -SE rule as before; plus1se = also a non-NULL freeze whose pooled check is not worse than +1 SE).
make_s53_v4.py (written 2026-10-12T16:31:30Z, lead 16:4xZ / 16:5xZ, radlab, before any of the round's data): = make_s53_v3.py with rule (c) OFF (no re-sized strength
arms) and escal OUT; rules (a) and (b) unchanged; the chain runs phase 1 only unless its PHASE2_ON file exists.
v3 (10-12 07:5xZ, engineer 07:5xZ: his DLY 'scarry' route reads s_carry per layer from the .pt, certified 10-11; exporter v6 accepts it):
s_carry moves from ALL_ONLY to PER_LAYER. Otherwise = v2. Export: export_dials_s46_v6.py.
v2 (10-12 07:5xZ, lead 07:5xZ + the engineer's route list of 07:4xZ, written before any S53 data): the dial families of rule (b) follow the
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
PER_LAYER = ('iso', 'nfs', 'nkl1', 'nkl2', 'nkl3', 'k4la1', 'kc', 's_carry')      # v2: engineer 07:4xZ, certified per layer; v3: + s_carry (exporter v6)
ALL_ONLY = ('bscale', 'X1', 'ssb', 'k4', 'nfd', 'n6b', 'n6r', 'n6d')      # v2: one global shift only (v3: s_carry moved to PER_LAYER)
PENDING = {'k21n': 'dk21n_7_10', 'rsy': 'drsy_11_14'}      # v2: per layer iff the engineer's S51 parity line says PASS; out if FAILED; else 'all'
ENG = [l for l in Path('../TEAM_LOG.md').read_text().splitlines() if l.startswith('[engineer] ')]
ROUTES = read('cert/RAD/ROUTES.json') if Path('cert/RAD/ROUTES.json').exists() else None      # route table (data, make_routes_table_v1.py)
if ROUTES:
    PER_LAYER, ALL_ONLY, PENDING = tuple(ROUTES['per_layer']), tuple(ROUTES['all_only']), dict(ROUTES['pending'])
BANDS = {k: [tuple(b) for b in v] for k, v in (ROUTES or {}).get('bands', {}).items()}
DEFAULT_BANDS = [(1, 6), (7, 10), (11, 14), (0, 15)]


def route(fam):
    if fam in BANDS:
        return 'bands'
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


CRULE = (Path('cert/RAD/CONTINUE_RULE').read_text().split() or ['se'])[0] if Path('cert/RAD/CONTINUE_RULE').exists() else 'se'


def accepted(rnd):      # the round CONTINUES: its frozen point may start the next round (make_continue_rule_v1.py)
    files = [Path('cert/RAD/%s/%s' % (rnd, f)) for f in ('CHECK_VERDICT_v1.json', 'RESERVE_VERDICT_v1.json', 'SECOND_VERDICT_v1.json')]
    if any(p.exists() and read(p).get('accepted') is True for p in files):
        return True
    if CRULE != 'plus1se':
        return False
    fp = Path('cert/RAD/%s/FROZEN_POINT_v1.json' % rnd)
    if not fp.exists() or read(fp).get('null'):
        return False
    for name in ('RESERVE_VERDICT_v1.json', 'SECOND_VERDICT_v1.json'):
        p = Path('cert/RAD/%s/%s' % (rnd, name))
        if p.exists():
            q = read(p)['pooled']
            return q['pooled_pct'] <= q['se_pct']
    p = Path('cert/RAD/%s/CHECK_VERDICT_v1.json' % rnd)
    if not p.exists():
        return False
    c = read(p)['pt_vs_ab']
    return c['pooled_pct'] <= c['se_pct']


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
    drop = ['erest', 'h20', 'h6', 'escal'] + sorted(a for a, v in last.items() if v >= 0 and a not in ('erest', 'h20', 'h6', 'escal'))      # escal OUT (lead 16:5xZ)
    cfg = dict(name='S53', note='trainer RAD S53 (lead 10-12 07:4xZ): start = %s frozen point; arms by the rules of make_s53.py; sizing A48 + A50 '
                                '(+ A51 + A53 under the 12 h rule), check A52 7101288-319.' % src,
               patch={'ss': -1, 'ssb': 1, 'rdc': 2.0}, params=f['params'], hu=f['head'], bill_saving_pct=0.0, drop_arms=','.join(drop),
               sizing=SIZING, sizing_source='fresh', check_seeds=[7101288, 7101319], decide='ab', check_nd=False)
    sys.path.insert(0, 'cert/RAD')
    from absorb_v1 import absorb, post_absorbed      # cert/RAD/ABSORB.json (lead 18:3xZ): absorbed card deltas join the patch
    cfg, absorbed_now = absorb(cfg, 53)
    out.write_text(json.dumps(cfg, indent=1))
    post_absorbed(cfg, absorbed_now, 53)
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
        for lo, hi in (BANDS[fam] if routes[fam] == 'bands' else DEFAULT_BANDS if routes[fam] == 'layer' else [(0, 15)]):
            s = G[:, lo:hi + 1].sum(1); ta, tb, tt = t(s[:half]), t(s[half:]), t(s)
            if tt * tt > 6 and ta * tb > 0 and min(abs(ta), abs(tb)) > 1.5:
                dials.append(fam if (lo, hi) == (0, 15) else '%s:%d:%d' % (fam, lo, hi))
    DG = HOME / 'arms_grad_v1'
    subprocess.run([PY, '-B', RAD, 'build', str(HOME / 'config.json'), str(DG), ','.join(dials) or '-', '-'], check=True)   # (a) + (b)
    src = 'S52' if Path('cert/RAD/S52/solve_v1.json').exists() else 'S51'      # (c) strength arms with single-arm LOO < 0
    fxp = Path('cert/RAD/%s/arms_fixed_v1/arms.json' % src)      # v4: S52 v2 has no fixed arms (rule (c) off there too)
    solve, sizes = read('cert/RAD/%s/solve_v1.json' % src), (read(fxp)['size'] if fxp.exists() else {})
    failed = [a for a in sizes if any('%s PORT PARITY %s: FAILED' % (src, a) in l for l in ENG)]      # v2: arms whose port failed
    sys.path.insert(0, 'cert/RAD')
    from absorb_v1 import rule_c_once, strength_name_ok      # fagg fix + rule (c) once (fix_make_absorb_v1.py)
    strength = [a for a, v in solve['arm_alone_loo_pct'].items() if v < 0 and a in sizes and not a.startswith('fkappa') and strength_name_ok(a) and a not in failed]
    spec = []
    for a in strength:
        fam, lo, hi = a[1:].rsplit('_', 2)
        spec.append('%s:%s:%s:%s' % (fam, lo, hi, sizes[a]))
    strength, spec = [], []      # rule (c) OFF (lead 16:5xZ, radlab: the previous round's strength arms are spent)
    if cfg.get('rule_c_once'):      # ... except ONCE in the round that absorbs a card delta (cert/RAD/ABSORB.json; radlab report v1 item 3)
        src, strength, spec = rule_c_once(53, cfg['rule_c_once'])
    DF = HOME / 'arms_fixed_v1'
    if spec:
        subprocess.run([PY, '-B', RAD, 'fixed', str(HOME / 'config.json'), str(DF), ','.join(spec)], check=True)
    agg, agg_note = {}, ''      # (d) ONE aggregate weak-band arm 'fagg' from radlab's frozen builder (cert/RAD/make_agg_arm_v2.py)
    import hashlib
    AGG_SCRIPT, AGG_SHA = Path('../radlab/aggregate_spec_v1.py'), 'd0ab129b619a0fbcf66d2f5c254ac1a481f2d15035ca98aa5604f08ac6369b0b'
    PTLOAD, PTLOAD_SHA = Path('../radlab/ptload.py'), '0eb755da33771220094a4ecd37219835f446811bfcb4c3db066627f12dd5ce0d'      # the builder imports it
    aggf, prevk = HOME / 'aggregate_spec_v1.json', Path('cert/RAD/S%d/keep_v1.json' % (int(HOME.name[1:]) - 1))
    if spec:
        agg_note = 'none: rule (c) arms present'
    elif not AGG_SCRIPT.exists() or hashlib.sha256(AGG_SCRIPT.read_bytes()).hexdigest() != AGG_SHA or not PTLOAD.exists() \
            or hashlib.sha256(PTLOAD.read_bytes()).hexdigest() != PTLOAD_SHA:
        agg_note = 'none: radlab/aggregate_spec_v1.py or radlab/ptload.py missing or not the frozen sha'
    else:
        if not aggf.exists():
            subprocess.run([PY, '-B', str(AGG_SCRIPT), 'run_RAD_%s_gen/results' % HOME.name, 'a_R%sg' % HOME.name, str(aggf)] + ([str(prevk)] if prevk.exists() else []), check=True)
        sp = read(aggf).get('spec') or ''
        comps = [x.split(':') for x in sp.split(',') if x]
        agg_note = 'radlab/aggregate_spec_v1.py (sha %s...): %s' % (AGG_SHA[:16], sp or 'no weak band: no arm')
        if comps:
            sys.path.insert(0, '.')
            from otr_common import load_params, load_pl, save_params
            A0, d0 = load_params(torch, cfg['params']); pl = {k: [float(x) for x in v] for k, v in load_pl(torch, cfg['params']).items()}
            for fam, lo, hi, sh in comps:
                lo, hi, sh = int(lo), int(hi), float(sh)
                pl[fam] = [x + (sh if lo <= l <= hi else 0.0) for l, x in enumerate(pl.get(fam, [0.0] * 16))]
                agg[fam] = (lo, hi, sh)
            DF.mkdir(parents=True)
            save_params(torch, DF / 'arm_fagg.pt', A0, d0, pl); (DF / 'arm_fagg_hu.json').write_text(json.dumps(read(cfg['hu'])))
            (DF / 'arms.json').write_text(json.dumps(dict(live=['fagg'], size=dict(fagg=1.0), base=cfg['params'], base_hu=cfg['hu'], spec=sp,
                                                          source=agg_note, aggregate_spec=str(aggf)), indent=1))
    sys.path.insert(0, 'cert/RAD')
    from zero_arms_v1 import drop_zero_arms      # zero-arm guard (lead 18:1xZ): out of arms.json 'live' before any arm is queued
    zero = drop_zero_arms([d for d in (DG, DF) if (d / 'arms.json').exists()], cfg, 'S53')
    keep = read(DG / 'arms.json')['live'] + (read(DF / 'arms.json')['live'] if (DF / 'arms.json').exists() else [])
    stamp = utc()
    (HOME / 'keep_v1.json').write_text(json.dumps(dict(keep=keep, utc=stamp, dirs=[str(DG)] + ([str(DF)] if (DF / 'arms.json').exists() else [])), indent=1))
    (HOME / 'RULES_v1.json').write_text(json.dumps(dict(
        schema='trainer_RAD_rules_v1', utc=stamp, round='S53', written_before_any_arm_result=True, zero_arms_dropped=zero, absorbed=cfg.get('absorbed'), rule_c_once=cfg.get('rule_c_once'), start=cfg['params'], patch=cfg['patch'],
        sizing='phase 1 = A48 7101032-095 + A50 7101160-223 (128 fresh, 8M: truth-noise-corrected %); phase 2 = A51 7101224-287 + A53 7101352-415 '
               'only if the freeze projected from phase 1 lands within 12 h of the round start (chain_s53.sh), else freeze on phase 1',
        gradients='64 at the start point on A36 7100264-327', arms=keep, gradient_arms_dropped=cfg['drop_arms'],
        dial_rule='t^2 > 6 on 64 gradients with both halves of the same sign at |t| > 1.5; on the native routes only (make_s53_v3.py)',
        routes={f: r for f, r in sorted(routes.items()) if r}, route_table=ROUTES, dials=dials, aggregate_rule='(d) one arm fagg from radlab/aggregate_spec_v1.py (frozen sha d0ab129b...)', aggregate={f: dict(lo=v[0], hi=v[1], shift=v[2]) for f, v in agg.items()}, aggregate_note=agg_note, strength_port_failed=failed, strength_rule='OFF (lead 16:5xZ, radlab: re-sized strength arms are spent)',
        strength=strength, port_rule='cert/RAD/S53/PORT_RULE_v1.json',
        solve='ridge + leave-one-network-out (rad_v6.py freeze; candidates all / no dials / no fixed / fixed only); null if LOO >= 0',
        check='A52 7101288-319 (16M): frozen vs the unrefitted start, ACCEPT iff pooled change < -SE; sealed A52 7101320-351 = the physicist'), indent=1))
    print('zero arms dropped: %s' % (' '.join(zero) or 'none'))
    print('S53 arms (%d): %s | dials %s | routes %s | strength (%s) %s' % (len(keep), ' '.join(keep), dials, {f: r for f, r in sorted(routes.items()) if r}, src, strength))
