#!/usr/bin/env python3
"""TAG opus-physicist T151 = the CARRIER-DIAL scan on top of C (lead 18:5xZ GO; the law's and K31L4's leaves get NO gradient in RAD runs, otr_m_v7 REPLAY,
so no round has moved them since S48-S50 while gk, C and the RAD rounds changed the card). Screened on FRESH networks (lead: 7100584-591 hosted a dozen
screens today): C's reference on A41 7100608-623 (16M truths) in this run, each arm on the same 16, paired; a passing arm is confirmed on 7100592-607
against T141's C there. Arms (pl deltas on the S51 point's stored shifts; T141's C requests with 'rank' changed):
  (sp) / (sm)  the pair law's plane strength nfs +0.15 / -0.15 at L7-14 (k31 1.2406 -> 1.39 / 1.09 at L7-9, 1.0147 -> 1.16 / 0.86 at L10-14)
  (dp) / (dm)  the law's dscale nfd +0.15 / -0.15 at L1-14 (1.3087 -> 1.4587 / 1.1587: the multiple of C taken out of the plane)
  (kp)         the K31L4 plane x1.2 at L7-14, every age: nkl_i += 0.2 x (base_i + stored_i), bases nkl1 0 / nkl2 1 / nkl3 0
Served by scout-card's worker_sc8 from one queue (physicist/t151_carrier). 'screen' = the reference + the 5 arms on 7100608-623;
'confirm <arm> ...' = those arms on 7100592-607 (ids ph151x<arm>_r<seed>).  usage: t151_make.py screen | t151_make.py confirm <arm> [<arm>]"""
import json
import sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; RUN = HERE / 't151_carrier'
PARAMS = HERE.parent / 'trainer' / 'cert' / 'RAD' / 'S51' / 'point_v1' / 'params.pt'
torch.set_num_threads(1)
stored = {f: [float(x) for x in v] for f, v in torch.load(PARAMS, map_location='cpu', weights_only=True)['pl'].items()}
band = lambda lo, hi, f: [f(l) if lo <= l <= hi else 0.0 for l in range(16)]
ARMS = dict(sp={'nfs': band(7, 14, lambda l: 0.15)}, sm={'nfs': band(7, 14, lambda l: -0.15)},
            dp={'nfd': band(1, 14, lambda l: 0.15)}, dm={'nfd': band(1, 14, lambda l: -0.15)},
            kp={f: band(7, 14, lambda l, f=f, b=b: 0.2 * (b + stored[f][l])) for f, b in (('nkl1', 0.0), ('nkl2', 1.0), ('nkl3', 0.0))})


def request(seed, rid, arm):
    r = json.loads(sorted((HERE / 't141_k4l_ext' / 'claimed').glob('ph141_r7100584_c.w*.json'))[0].read_text())
    assert set(r['pl']) == {'k4la3', 'k4la4'} and r['params'] == str(PARAMS) and r['rank'] == 7100584
    r['rank'] = seed
    if arm:
        r['pl'].update(ARMS[arm])
    r['id'] = rid
    (RUN / 'queue' / ('%s.json' % rid)).write_text(json.dumps(r))


for d in ('claimed', 'queue', 'results'):
    (RUN / d).mkdir(parents=True, exist_ok=True)
(RUN / 'allowed_seeds.txt').write_text(','.join(map(str, range(7100592, 7100624))))
if sys.argv[1] == 'screen':
    assert not list((RUN / 'queue').iterdir()) and not list((RUN / 'results').iterdir())
    for s in range(7100608, 7100624):
        request(s, 'ph151_ref_r%d' % s, None)
    for arm in ('sp', 'sm', 'dp', 'dm', 'kp'):
        for s in range(7100608, 7100624):
            request(s, 'ph151%s_r%d' % (arm, s), arm)
        print('T151 arm %s: %s' % (arm, {f: [round(x, 5) for x in v][1:15] for f, v in ARMS[arm].items()}))
    print('T151 screen: 96 requests')
else:
    for arm in sys.argv[2:]:
        for s in range(7100592, 7100608):
            request(s, 'ph151x%s_r%d' % (arm, s), arm)
        print('T151 confirm arm %s: 16 requests (7100592-607)' % arm)
