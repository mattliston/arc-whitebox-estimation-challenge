"""make_s60_v16.py (written 2026-10-13T16:12:57Z, lead 15:5xZ): = make_s60_v15.py with the config stage on absorb_v3.py (+ start_pl_add deltas, e.g. a fired c0 literal)
(cert/RAD/fix_make_absorb3_v1.py). Arm stage unchanged.
make_s60_v15.py (written 2026-10-13T16:09:07Z, lead 15:5xZ): = make_s60_v14.py + the new-family bands c0 (K31 row factor) L8-14 and pe3 L8-11 as tier-(a) candidates
(cert/RAD/fix_make_c0pe3_v1.py). Config stage unchanged.
make_s60_v14.py (written 2026-10-13T15:53:28Z, lead 15:5xZ): = make_s60_v13.py + the new family band k4la3 / k4la4 at L4-7 (K4L ages 3 / 4) as a tier-(a) candidate
(cert/RAD/fix_make_k4l47_v1.py). Config stage unchanged.
make_s60_v13.py (written 2026-10-13T10:43:08Z, lead 10:3xZ): = make_s60_v12.py with the config stage on absorb_v2.py: an ABSORB.json start_pl delta (K4L ages 3-4) is written into
this round's start point (<round>/start_v1/params.pt) when the start point does not carry it yet. Arm stage unchanged.
make_s60_v12.py (written 2026-10-12T23:48:25Z, lead 23:3xZ): = make_s60_v11.py + the new family k21k (K21 scale on the kink rows, otr_pl_v2) on L5-9 / L10-11 / L12-14 as a tier-(a) candidate.
make_s60_v11.py (written 2026-10-12T23:34:39Z, lead 23:2xZ): = make_s60_v10.py + NEW DIAL FAMILIES ds / pe3 / pe4 per band as tier-(a) candidates (cert/RAD/fix_make_newfam_v1.py).
make_s60_v10.py (written 2026-10-12T23:04:49Z, lead 23:1xZ): = make_s60_v9.py with the LEAN arm stage (cert/RAD/fix_make_lean_v1.py; as make_s53_v12.py: n = 64 sizing on the
round's first block (the chain), at most 7 arms (a) > (b) > (c, absorbing round only, |t| >= 2, at most 2) > (d) in arms_agg_v1). Config stage unchanged.
make_s60_v9.py (written 2026-10-12T18:44:37Z, lead 18:3xZ): = make_s60_v8.py + the ABSORB hook (cert/RAD/absorb_v1.py; inert while cert/RAD/ABSORB.json is absent) + the fagg fix of the strength-arm name parse (cert/RAD/fix_make_absorb_v1.py). Nothing else changed.
make_s60_v8.py (written 2026-10-12T18:34:19Z, lead 18:1xZ): = make_s60_v7.py with the ZERO-ARM GUARD (cert/RAD/zero_arms_v1.py, cert/RAD/fix_make_zero_v1.py): an arm whose parameter file equals the start point bit for bit leaves arms.json 'live' before keep_v1.json and before any arm is queued; one TEAM_LOG line when any is dropped. Nothing else changed.
make_s60_v7.py (written 2026-10-12T17:13:27Z, radlab): = make_s60_v6.py with radlab/ptload.py (imported by the aggregate builder) pinned by sha256 as well.
make_s60_v6.py (written 2026-10-12T17:07:53Z, radlab 17:0xZ): = make_s60_v5.py with rule (d) taken from the frozen radlab aggregate builder (cert/RAD/make_agg_arm_v2.py).
make_s60_v5.py (written 2026-10-12T17:01:13Z, lead 17:0xZ / radlab): = make_s60_v4.py with rule (d), ONE aggregate weak-band arm fagg (cert/RAD/make_agg_arm_v1.py: per family
the largest-|t| routed band with t^2 >= 4 that rule (b) did not take and that overlaps none of its strict bands, moved against its gradient sign by
0.02 x min(1, |t|/6), all in one arm).
make_s60_v4.py (written 2026-10-12T16:51:14Z, lead / radlab 16:5xZ): = make_s60_v3.py with rule (b)'s routes read from cert/RAD/ROUTES.json at the arm stage (adds
per-family band lists: k4 1-7 / 8-11 / 12-14 / 15-15 / all); the strict rule unchanged.
make_s60_v3.py (written 2026-10-12T16:44:27Z, lead 16:5xZ / radlab): = make_s60_v2.py with the start point = the latest round that CONTINUES (cert/RAD/CONTINUE_RULE:
absent or se = accepted by the -SE rule as before; plus1se = also a non-NULL freeze whose pooled check is not worse than +1 SE).
make_s60_v2.py (written 2026-10-12T16:31:19Z, lead 16:4xZ / 16:5xZ, radlab, before any of the round's data): = make_s60.py with rule (c) OFF (no re-sized strength
arms) and escal OUT; rules (a) and (b) unchanged; the chain runs phase 1 only unless its PHASE2_ON file exists.
TAG opus-trainer RAD S60 (gen_round.py 60 (lead 07:5xZ: rounds keep the S54 template until 10-17), written 2026-10-12T08:03:37Z): the pre-declared,
rule-driven construction of round S60, run by cert/RAD/S60/chain_s60.sh.
  config   start = the latest ACCEPTED frozen point of S59 back to S50 (accepted = that round's CHECK, RESERVE or SECOND verdict file says so);
           the S48 patch {ss -1, ssb 1, rdc 2.0}; sizing on FRESH 8M blocks (truth-noise-corrected % via rad_v6.py): phase 1 = A90 7103720-7103783
           + A92 7103848-7103911, phase 2 = A93 7103912-7103975 + A95 7104040-7104103 (the chain's 12 h rule); check on A91's first half 7103784-7103815 (16M)
           vs the unrefitted start, ACCEPT iff pooled change < -SE; sealed = A91's second half 7103816-7103847 (the physicist). A94 7103976-7104039
           stays an unassigned reserve. Gradients: 64 at the start on A36 7100264-327, as every round.
  arms     after the gradients, by fixed rules (make_s53_v3.py's):
           (a) learned-array / head arms: cert/refit/jmix_generated_v7.py via rad_v6.py build with the Nf settings; dropped: erest, h20, h6
               (the default) and every gradient arm whose MOST RECENT single-arm LOO was >= 0 (the latest round with a solve, S59 back to S51);
           (b) band dial arms by S51's strict rule (t^2 > 6 on the 64 gradients, both halves of the same sign at |t| > 1.5; bands 1-6 / 7-10 /
               11-14 / all) on the native routes only: PER_LAYER / ALL_ONLY / PENDING as in make_s53_v3.py; kappa excluded;
           (c) strength arms: the forward arms of the latest round with a solve (S59 back to S51) with single-arm LOO < 0 on its sizing networks, rebuilt from the
               start with that round's trial shifts; kappa and any arm with an engineer line '<round> PORT PARITY <arm>: FAILED' excluded.
           writes keep_v1.json and RULES_v1.json before any arm result exists. The port rule is cert/RAD/S60/PORT_RULE_v1.json.
usage (from O/trainer): python -B cert/RAD/S60/make_s60.py config | arms"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HOME = Path('cert/RAD/S60')
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
SIZING = [[7103720, 7103783], [7103848, 7103911], [7103912, 7103975], [7104040, 7104103]]
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
    src = next(r for r in ('S59', 'S58', 'S57', 'S56', 'S55', 'S54', 'S53', 'S52', 'S51', 'S50') if accepted(r))
    f = read('cert/RAD/%s/FROZEN_POINT_v1.json' % src)
    last = {}
    for rnd in ('S51', 'S52', 'S53', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59'):      # the most recent single-arm LOO of each gradient arm: later rounds override earlier ones
        p = Path('cert/RAD/%s/solve_v1.json' % rnd)
        if p.exists():
            last.update({a: v for a, v in read(p)['arm_alone_loo_pct'].items() if a in GRAD_ARMS})
    drop = ['erest', 'h20', 'h6', 'escal'] + sorted(a for a, v in last.items() if v >= 0 and a not in ('erest', 'h20', 'h6', 'escal'))      # escal OUT (lead 16:5xZ)
    cfg = dict(name='S60', note='trainer RAD S60 (gen_round.py): start = %s frozen point; arms by the rules of make_s60.py; sizing A90 + A92 '
                                '(+ A93 + A95 under the 12 h rule), check A91 7103784-7103815.' % src,
               patch={'ss': -1, 'ssb': 1, 'rdc': 2.0}, params=f['params'], hu=f['head'], bill_saving_pct=0.0, drop_arms=','.join(drop),
               sizing=SIZING, sizing_source='fresh', check_seeds=[7103784, 7103815], decide='ab', check_nd=False)
    sys.path.insert(0, 'cert/RAD')
    from absorb_v3 import absorb, post_absorbed      # cert/RAD/ABSORB.json: patch + start_pl + start_pl_add deltas (v16, lead 15:5xZ)
    cfg, absorbed_now = absorb(cfg, 60, home=HOME)
    out.write_text(json.dumps(cfg, indent=1))
    post_absorbed(cfg, absorbed_now, 60)
    print('S60 config: start = %s frozen point (%s); gradient arms dropped: %s' % (src, f['params'], ','.join(drop)))
elif mode == 'arms':      # LEAN (lead 23:1xZ, radlab's plan as in make_s53_v12): sizing n = 64 on the round's first block; at most 7 arms, (a) > (b) > (c) > (d)
    import hashlib
    import torch
    sys.path.insert(0, '.'); sys.path.insert(0, 'cert/RAD')
    from otr_common import load_params, load_pl, save_params
    from zero_arms_v1 import drop_zero_arms
    from absorb_v1 import rule_c_once
    NAME, N = HOME.name, int(HOME.name[1:])
    cfg = read(HOME / 'config.json')
    seeds = [int(x) for x in Path('cert/Nf/gen_allow_grad_v1.txt').read_text().split(',')]
    rows = [torch.load('run_RAD_%s_gen/results/a_R%sg_r%d.pt' % (NAME, NAME, r), map_location='cpu', weights_only=False) for r in seeds]
    assert len(rows) == 64
    half = 32
    t = lambda x: float(x.mean() / (x.std(ddof=1) / len(x) ** .5)) if x.std() > 0 else 0.0
    G_of = lambda fam: np.stack([r['pl_grads'][fam].double().numpy() * (1e8 / r['scale']) for r in rows])
    routes = {fam: route(fam) for fam in rows[0]['pl_grads']}
    dials, dial_t = [], {}
    for fam in sorted(rows[0]['pl_grads'], key=str.lower):      # (b) band dial arms, strict rule, native routes only (ROUTES.json)
        if routes[fam] is None or fam == 'kappa':
            continue
        G = G_of(fam)
        if not np.abs(G).max() > 0:
            continue
        for lo, hi in (BANDS[fam] if routes[fam] == 'bands' else DEFAULT_BANDS if routes[fam] == 'layer' else [(0, 15)]):
            s = G[:, lo:hi + 1].sum(1); ta, tb, tt = t(s[:half]), t(s[half:]), t(s)
            if tt * tt > 6 and ta * tb > 0 and min(abs(ta), abs(tb)) > 1.5:
                name = fam if (lo, hi) == (0, 15) else '%s:%d:%d' % (fam, lo, hi)
                dials.append(name); dial_t['d' + name.replace(':', '_')] = round(tt, 2)
    NEWFAM = dict({f: [(5, 9), (10, 11), (12, 14), (15, 15)] for f in ('ds', 'pe3', 'pe4')}, k21k=[(5, 9), (10, 11), (12, 14)], k4la3=[(4, 7)], k4la4=[(4, 7)],
                  c0=[(8, 14)])      # newfam: + k21k (v12), + K4L ages 3 / 4 at L4-7 (v14), + c0 at L8-14 and pe3 at L8-11 (v15)
    NEWFAM['pe3'] = NEWFAM['pe3'] + [(8, 11)]
    new_t, new_dials = {}, []
    for fam, bands in NEWFAM.items():
        if fam not in rows[0]['pl_grads']:
            continue
        G = G_of(fam)
        for lo, hi in bands:
            s = G[:, lo:hi + 1].sum(1); ta, tb, tt = t(s[:half]), t(s[half:]), t(s)
            key = '%s:%d:%d' % (fam, lo, hi)
            ok = tt * tt > 6 and ta * tb > 0 and min(abs(ta), abs(tb)) > 1.5
            new_t[key] = dict(t=round(tt, 2), halves=[round(ta, 2), round(tb, 2)], admitted=bool(ok))
            if ok:
                new_dials.append(key); dial_t['d' + key.replace(':', '_')] = round(tt, 2)
    new_arms = {'d' + k.replace(':', '_') for k in new_dials}
    DG = HOME / 'arms_grad_v1'
    subprocess.run([PY, '-B', RAD, 'build', str(HOME / 'config.json'), str(DG), ','.join(dials + new_dials) or '-', '-'], check=True)   # (a) + new families + (b)
    rule_c, gate, rc_src = [], {}, None      # (c) OFF, except ONCE in a round that absorbs a card delta (cert/RAD/ABSORB.json): the rule (c)
    if cfg.get('rule_c_once'):               # candidates (absorb_v1.rule_c_once) kept iff |t| >= 2 on THIS round's band-summed gradient, at most 2 (lead 23:1xZ)
        rc_src, cands, _ = rule_c_once(N, cfg['rule_c_once'])
        sizes = read('cert/RAD/%s/arms_fixed_v1/arms.json' % rc_src)['size'] if rc_src else {}
        for arm in cands:
            fam, lo, hi = arm[1:].rsplit('_', 2); lo, hi = int(lo), int(hi)
            tt = t(G_of(fam)[:, lo:hi + 1].sum(1)) if fam in rows[0]['pl_grads'] else 0.0
            gate[arm] = dict(family=fam, lo=lo, hi=hi, t=round(tt, 2), size=sizes.get(arm), passes=abs(tt) >= 2)
        rule_c = sorted((a for a, g in gate.items() if g['passes']), key=lambda a: -abs(gate[a]['t']))[:2]
    spec = ['%s:%d:%d:%s' % (gate[a]['family'], gate[a]['lo'], gate[a]['hi'], gate[a]['size']) for a in rule_c]
    DF = HOME / 'arms_fixed_v1'
    if spec:
        subprocess.run([PY, '-B', RAD, 'fixed', str(HOME / 'config.json'), str(DF), ','.join(spec)], check=True)
    agg, agg_note, DA = {}, '', HOME / 'arms_agg_v1'      # (d) ONE aggregate weak-band arm 'fagg' from radlab's frozen builder, in its own dir
    AGG_SCRIPT, AGG_SHA = Path('../radlab/aggregate_spec_v1.py'), 'd0ab129b619a0fbcf66d2f5c254ac1a481f2d15035ca98aa5604f08ac6369b0b'
    PTLOAD, PTLOAD_SHA = Path('../radlab/ptload.py'), '0eb755da33771220094a4ecd37219835f446811bfcb4c3db066627f12dd5ce0d'
    aggf, prevk = HOME / 'aggregate_spec_v1.json', Path('cert/RAD/S%d/keep_v1.json' % (N - 1))
    if not AGG_SCRIPT.exists() or hashlib.sha256(AGG_SCRIPT.read_bytes()).hexdigest() != AGG_SHA or not PTLOAD.exists() \
            or hashlib.sha256(PTLOAD.read_bytes()).hexdigest() != PTLOAD_SHA:
        agg_note = 'none: radlab/aggregate_spec_v1.py or radlab/ptload.py missing or not the frozen sha'
    else:
        if not aggf.exists():
            subprocess.run([PY, '-B', str(AGG_SCRIPT), 'run_RAD_%s_gen/results' % NAME, 'a_R%sg' % NAME, str(aggf)] + ([str(prevk)] if prevk.exists() else []), check=True)
        sp = read(aggf).get('spec') or ''
        comps = [x.split(':') for x in sp.split(',') if x]
        agg_note = 'radlab/aggregate_spec_v1.py (sha %s...): %s' % (AGG_SHA[:16], sp or 'no weak band: no arm')
        if comps:
            A0, d0 = load_params(torch, cfg['params']); pl = {k: [float(x) for x in v] for k, v in load_pl(torch, cfg['params']).items()}
            for fam, lo, hi, sh in comps:
                lo, hi, sh = int(lo), int(hi), float(sh)
                pl[fam] = [x + (sh if lo <= l <= hi else 0.0) for l, x in enumerate(pl.get(fam, [0.0] * 16))]
                agg[fam] = (lo, hi, sh)
            DA.mkdir(parents=True)
            save_params(torch, DA / 'arm_fagg.pt', A0, d0, pl); (DA / 'arm_fagg_hu.json').write_text(json.dumps(read(cfg['hu'])))
            (DA / 'arms.json').write_text(json.dumps(dict(live=['fagg'], size=dict(fagg=1.0), base=cfg['params'], base_hu=cfg['hu'], spec=sp,
                                                          source=agg_note, aggregate_spec=str(aggf)), indent=1))
    dirs = [d for d in (DG, DF, DA) if (d / 'arms.json').exists()]
    zero = drop_zero_arms(dirs, cfg, NAME)      # zero-arm guard (lead 18:1xZ)
    live = {d: read(d / 'arms.json')['live'] for d in dirs}
    a_arms = [a for a in live[DG] if not (a.startswith('d') and a in dial_t)]
    a_arms += sorted((a for a in live[DG] if a in new_arms), key=lambda a: -abs(dial_t[a]))      # newfam: inside tier (a), by |t|
    b_arms = sorted((a for a in live[DG] if a.startswith('d') and a in dial_t and a not in new_arms), key=lambda a: -abs(dial_t[a]))
    c_arms = [a for a in rule_c if DF in live and a in live[DF]]
    d_arms = list(live.get(DA, []))
    order = a_arms + b_arms + c_arms + d_arms
    keep, capped = order[:7], order[7:]      # the cap (lead 22:5xZ / 23:1xZ): 7 arms
    stamp = utc()
    for d in dirs:
        if any(a in capped for a in live[d]):
            j = d / 'arms.json'
            shutil.copy2(j, d / ('arms.json.archived_%s_cap7' % stamp.replace('-', '').replace(':', '')))
            x = read(j); x['live'] = [a for a in x['live'] if a in keep]; x['capped_out'] = [a for a in live[d] if a in capped]
            j.write_text(json.dumps(x, indent=1))
    dirs = [d for d in dirs if read(d / 'arms.json')['live']]
    (HOME / 'keep_v1.json').write_text(json.dumps(dict(keep=keep, utc=stamp, dirs=[str(d) for d in dirs]), indent=1))
    (HOME / 'RULES_v1.json').write_text(json.dumps(dict(
        schema='trainer_RAD_rules_v1', utc=stamp, round=NAME, version='%s (lean, lead 23:1xZ)' % Path(sys.argv[0]).name, written_before_any_arm_result=True,
        zero_arms_dropped=zero, absorbed=cfg.get('absorbed'), rule_c_once=cfg.get('rule_c_once'), start=cfg['params'], patch=cfg['patch'],
        sizing='n = 64: base + every kept arm on %d-%d (the round\'s first sizing block); no phase 2; zero arms -> NULL at once' % tuple(SIZING[0]),
        gradients='64 at the start point on A36 7100264-327', arms=keep, capped_out=capped,
        priority='(a) gradient arms (jmix order) > (b) strict-rule bands (largest |t| first) > (c) absorbing round only, gated, at most 2 > (d) aggregate; cap 7',
        gradient_arms_dropped=cfg['drop_arms'], dial_rule='t^2 > 6 on 64 gradients with both halves of the same sign at |t| > 1.5; native routes (ROUTES.json)',
        dials_t=dial_t, routes={f: r for f, r in sorted(routes.items()) if r}, route_table=ROUTES,
        new_families_t=new_t, new_family_rule='ds / pe3 / pe4 on L5-9 / L10-11 / L12-14 / L15, k21k on L5-9 / L10-11 / L12-14, k4la3 / k4la4 on L4-7, c0 on L8-14, pe3 also on L8-11: strict rule; admitted bands sit in tier (a) after the jmix arms, largest |t| first',
        rule_c_source=rc_src, rule_c_gate=gate, rule_c_kept=rule_c, rule_c_rule='absorbing round only: rule (c) candidates kept iff |t| >= 2 on this round\'s band-summed gradient; at most 2, largest |t| first',
        aggregate_rule='(d) one arm fagg from radlab/aggregate_spec_v1.py (frozen sha d0ab129b...), arms_agg_v1', aggregate={f: dict(lo=v[0], hi=v[1], shift=v[2]) for f, v in agg.items()},
        aggregate_note=agg_note, port_rule='cert/RAD/%s/PORT_RULE_v1.json' % NAME,
        solve='ridge + leave-one-network-out (rad_v6.py freeze on config_n64.json; candidates all / no dials / no fixed / fixed only); null if LOO >= 0'), indent=1))
    tab = '; '.join('%s t %+.2f %s' % (a, g['t'], 'PASS' if g['passes'] else 'no') for a, g in gate.items()) or 'not an absorbing round: rule (c) off'
    with open('../TEAM_LOG.md', 'a') as log:
        log.write('[trainer] %s %s ARMS LEAN PRE-REGISTERED (lead 23:1xZ; keep_v1.json / RULES_v1.json written before any arm result): %d arms %s (capped out: %s); '
                  '(b) dials t %s; NEW FAMILIES t (band: t, halves, admitted) %s; (c) %s -> kept %s; (d) %s; zero arms dropped: %s; sizing n = 64 on %d-%d.\n'
                  % (stamp, NAME, len(keep), ' '.join(keep) or 'NONE', ' '.join(capped) or 'none', dial_t or '{}',
                     '; '.join('%s %+.2f (%+.2f / %+.2f)%s' % (k, v['t'], v['halves'][0], v['halves'][1], ' IN' if v['admitted'] else '') for k, v in new_t.items()) or 'none in the gradients',
                     tab, rule_c or 'none', agg_note or 'none',
                     ' '.join(zero) or 'none', SIZING[0][0], SIZING[0][1]))
    print('%s arms lean (%d): %s | capped out: %s | dials %s | rule (c) kept %s | aggregate %s' % (NAME, len(keep), ' '.join(keep) or 'NONE', ' '.join(capped) or 'none', dial_t, rule_c, agg_note))
