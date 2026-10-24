"""TAG opus-trainer RAD (10-12 17:0xZ; lead 16:5xZ: a lead-only PROMOTE that needs no change to the other loops): one pass over the rounds S52..S66.
For a round whose directory holds LEAD_PROMOTE (the lead's one-line reason) and no LEAD_PROMOTE.done / .refused marker:
  - waits (does nothing yet) until the round has FROZEN_POINT_v1.json and CHECK_VERDICT_v1.json;
  - REFUSES (marker .refused + one TEAM_LOG line) if the freeze is NULL, if RESERVE_VERDICT_v1.json already exists (never overwritten), or if the
    round does not CONTINUE (pooled frozen-vs-start check worse than +1 SE); a round already 'accepted' by its CHECK needs nothing (marker .done);
  - otherwise writes the round's standard verdict file RESERVE_VERDICT_v1.json with accepted = true (the field the physicist's stage 2, the
    engineer's port and therefore release key on), the lead's reason, the round's pooled check numbers, and the cumulative LOO since the last fired
    point (sum of the frozen LOO of every round since that fire whose point the line continued from); marker .done + one TEAM_LOG line.
The last fired round M = the newest O/fire/s<M>s_<date> with receipt.json (the physicist's stage-2 baseline rule).
usage (from O/trainer): python -B cert/RAD/lead_promote_v1.py   (cert/RAD/lead_promote_v1.sh calls it once a minute)"""
import json
import re
import time
from pathlib import Path

read = lambda p: json.loads(Path(p).read_text())
stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
log = lambda line: open('../TEAM_LOG.md', 'a').write('[trainer] %s %s\n' % (stamp, line))
fired = [d for d in Path('../fire').iterdir() if re.fullmatch(r's\d+s_\d{8}', d.name) and (d / 'receipt.json').exists()]
M = int(re.match(r's(\d+)s_', max(fired, key=lambda d: (d / 'receipt.json').stat().st_mtime).name).group(1)) if fired else 50


def continues(n):
    h = Path('cert/RAD/S%d' % n)
    for name in ('CHECK_VERDICT_v1.json', 'RESERVE_VERDICT_v1.json', 'SECOND_VERDICT_v1.json'):
        if (h / name).exists() and read(h / name).get('accepted') is True:
            return True
    if not (h / 'FROZEN_POINT_v1.json').exists() or read(h / 'FROZEN_POINT_v1.json').get('null') or not (h / 'CHECK_VERDICT_v1.json').exists():
        return False
    c = read(h / 'CHECK_VERDICT_v1.json')['pt_vs_ab']
    return c['pooled_pct'] <= c['se_pct']


for n in range(52, 67):
    h = Path('cert/RAD/S%d' % n)
    if not (h / 'LEAD_PROMOTE').exists() or (h / 'LEAD_PROMOTE.done').exists() or (h / 'LEAD_PROMOTE.refused').exists():
        continue
    if not (h / 'FROZEN_POINT_v1.json').exists() or not (h / 'CHECK_VERDICT_v1.json').exists():
        continue      # not frozen / checked yet: wait
    reason = (h / 'LEAD_PROMOTE').read_text().strip()[:300]
    f, v = read(h / 'FROZEN_POINT_v1.json'), read(h / 'CHECK_VERDICT_v1.json')
    if v.get('accepted') is True:
        (h / 'LEAD_PROMOTE.done').write_text(stamp + ' already accepted by its CHECK\n')
        log('S%d LEAD_PROMOTE: nothing to do, the round is already accepted by its check (the port / stage 2 run as usual).' % n)
        continue
    why = 'its freeze is NULL' if f.get('null') else 'RESERVE_VERDICT_v1.json already exists' if (h / 'RESERVE_VERDICT_v1.json').exists() \
        else None if continues(n) else 'it does not continue (pooled check worse than +1 SE)'
    if why:
        (h / 'LEAD_PROMOTE.refused').write_text(stamp + ' ' + why + '\n')
        log('S%d LEAD_PROMOTE REFUSED: %s (cert/RAD/lead_promote_v1.py; the reason given: %s).' % (n, why, reason))
        continue
    line_rounds = [k for k in range(M + 1, n + 1) if (Path('cert/RAD/S%d/FROZEN_POINT_v1.json' % k).exists() and continues(k))]
    cum = sum(read('cert/RAD/S%d/FROZEN_POINT_v1.json' % k)['loo_pct'] for k in line_rounds)
    c = v['pt_vs_ab']
    out = dict(utc=stamp, round='S%d' % n, accepted=True, rule='LEAD_PROMOTE (lead 16:5xZ): the lead promoted a continued round to the downstream '
               '(stage 2 vs the last fired point, port, release gates); the -SE check itself was not met', lead_promote=dict(reason=reason, file=str(h / 'LEAD_PROMOTE')),
               pooled=dict(pooled_pct=c['pooled_pct'], se_pct=c['se_pct'], better=c.get('better')), check_verdict=str(h / 'CHECK_VERDICT_v1.json'),
               frozen_loo_pct=f['loo_pct'], last_fired_round='S%d' % M, rounds_continued_since_last_fire=['S%d' % k for k in line_rounds],
               cumulative_loo_since_last_fire_pct=cum)
    p = h / 'RESERVE_VERDICT_v1.json'
    with p.open('x') as fh:
        json.dump(out, fh, indent=1)
    (h / 'LEAD_PROMOTE.done').write_text(stamp + ' promoted\n')
    log('S%d LEAD_PROMOTE DONE (reason: %s): trainer/cert/RAD/S%d/RESERVE_VERDICT_v1.json accepted = true (field lead_promote); check %+.3f%% (se %.3f); '
        'cumulative LOO since the last fired S%d over %s = %+.3f%%. The physicist\'s stage 2 (vs the last fired point), the engineer\'s port and release '
        'run on it as on any accepted round.' % (n, reason, n, c['pooled_pct'], c['se_pct'], M, ' '.join(out['rounds_continued_since_last_fire']), cum))
