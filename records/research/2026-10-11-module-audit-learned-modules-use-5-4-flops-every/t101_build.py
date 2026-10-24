#!/usr/bin/env python3
"""TAG opus-physicist T101 build: MODULE AUDIT at S47e = the trainer's frozen Nd point on the v11 cut simulator (lead 10-11 RESTART), seeds 7100136-151,
paired against the fired point (arm A of physicist/t93_tailscan). Each arm switches ONE module / term / learned part off (or to x0.5 where off breaks
the card) through the request: per-layer dial shifts 'pl' (added to the checkpoint's stored shifts; values = -(base + stored) give 0), the ncar list,
or the module fields. Effective values from the Nd checkpoint (cert/refit/jmNd_p1.pt) and the bases in otr_pl.py / otr_m.py docstrings.
usage: t101_build.py <run dir> <workers> <arm> [<arm> ...]   (arm names: see ARMS; '<name>@<lo>-<hi>' restricts a pl arm to layers lo..hi)"""
import copy, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
RUN, NW, NAMES = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3:]
base = json.loads((HERE / 't92_Nd_candidate_request_v1.json').read_text())
for k in ('id', 'rank', 'scale', 'lossw', 'truth_plan'): base.pop(k, None)
base.update(mode='eval', keep_all=False, keep_final=False)
L = range(16)
k4la1 = [-0.86 if not 10 <= l <= 14 else -0.738 for l in L]
nfs_stored = [0.0] + [0.829] * 4 + [0.395] * 5 + [0.181] * 5 + [0.0]
kc_stored = [0.0] + [0.574] * 9 + [0.544] * 5 + [0.0]
iso_eff = [-0.094 if l < 10 else -0.064 for l in L]
ARMS = {   # name -> dict(pl={family: 16 shifts}) and / or request edits
    'F4rule0': dict(pl=dict(f4a1=[-0.656] * 16, f4a2=[-0.6] * 16)),
    'F4S2_0': dict(pl=dict(x2a1=[-0.444] * 16, x2a2=[-0.171] * 16)),
    'K4L0': dict(pl=dict(k4la1=k4la1, k4la2=[-0.86] * 16)),
    'LAWfold0': dict(pl=dict(n6b=[-1.142] * 16, n6r=[-0.697] * 16, n6d=[-0.791] * 16)),
    'LAWk31_0': dict(pl=dict(nfs=[-(1.0 + s) for s in nfs_stored])),
    'K31L0': dict(pl=dict(nkl2=[-0.36] * 16, nkl3=[-0.64] * 16)),
    'PBASE0': dict(ncar_bsk=0.0),
    'EDGE0': dict(pl=dict(edge=[-0.873] * 16)),
    'HCOL0': dict(req=dict(hf=0)),
    'B6_0': dict(req=dict(b6=0)),
    'KC0': dict(pl=dict(kc=[-s for s in kc_stored])),
    'COVKNOB0': dict(pl=dict(iso=[-e for e in iso_eff], k21n=[0.048] * 16, k31=[-0.155] * 16)),
    'BSRC0': dict(pl=dict(bsrc=[-0.965] * 16)),
    'K22_0': dict(pl=dict(k22=[-1.0] * 16)),
    'HEADx05': dict(pl=dict(ds=[-0.392] * 15 + [-0.402])),
    'ROWF0': dict(pl=dict(c0=[-0.522] * 16, c1=[-0.5027] * 16)),
    'STAR0': dict(opt=dict(t102='zero')),          # needs physicist/t102_worker_star.py
    'STARX': dict(opt=dict(t102='exact')),         # needs physicist/t102_worker_star.py
    'STAR32': dict(req=dict(ss=None)),             # the card's rank-32 factored star instead of the row-sum star (ss 0)
    'PBASE_E': dict(ncar_pbase=dict(lo=7, hi=14)), # pbase's bsk only at 7-14 (early band off)
    'PBASE_L': dict(ncar_pbase=dict(lo=1, hi=10)), # pbase's bsk only at 1-10 (late band off)
}
RUN.mkdir()
for d in ('claimed', 'queue', 'results'): (RUN / d).mkdir()
seeds = list(range(7100136, 7100152))
(RUN / 'allowed_seeds.txt').write_text(','.join(map(str, seeds)) + '\n')
n = 0
for i, s in enumerate(seeds):
    for name in NAMES:
        key, _, band = name.partition('@'); a = ARMS[key]
        r = copy.deepcopy(base); r.update(id='t101_%d_%s' % (s, name.replace('@', 'L').replace('-', '_')), rank=s)
        if 'pl' in a:
            lo, hi = (int(x) for x in band.split('-')) if band else (0, 15)
            r['pl'] = {f: [v if lo <= l <= hi else 0.0 for l, v in enumerate(vals)] for f, vals in a['pl'].items()}
        for k, v in a.get('req', {}).items():
            if v is None: r.pop(k, None)
            else: r[k] = v
        if 'opt' in a:
            r['cutsim'] = dict(r['cutsim'], opts=dict(r['cutsim']['opts'], **a['opt']))
        if 'ncar_pbase' in a:
            r['ncar'] = [dict(e, **a['ncar_pbase']) if e.get('module') == 'pbase' else e for e in r['ncar']]
        if 'ncar_bsk' in a:
            r['ncar'] = [dict(e, bsk=a['ncar_bsk']) if e.get('module') == 'pbase' else e for e in r['ncar']]
        (RUN / 'claimed' / ('%s.w%d.json' % (r['id'], n % NW))).write_text(json.dumps(r)); n += 1
(RUN / 'STOP').write_text('')
print('requests', n, 'arms', NAMES)
