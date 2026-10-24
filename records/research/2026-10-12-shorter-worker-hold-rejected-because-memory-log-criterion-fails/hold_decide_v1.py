"""TAG opus-trainer RAD (10-12 13:3xZ; lead 13:3xZ; rule pre-declared in TEAM_LOG 13:29:03Z): decides cert/RAD/HOLD_RULE from S51's downstream in
cert/RAD/memlog_v1.jsonl. A minute counts as RELEASE-ONLY when release's python processes hold > 0.5 GiB of RSS while the engineer's hold <= 0.5 GiB
(no port / native run) and the physicist's <= 1.0 GiB (no stage-2 judge; the stage-2 loop itself is tiny). Rule: 'ready_table' iff there are
>= 10 release-only minutes, memory_pressure free never drops below 60% in them, and swap rises at most 64 MiB within them;
otherwise 'release' stays. Writes the decision (a new HOLD_RULE content only if it changes; the old one is archived) and one TEAM_LOG line with
the minima of every phase. usage (from O/trainer): python -B cert/RAD/hold_decide_v1.py"""
import json
import os
import time
from pathlib import Path

rows = [json.loads(l) for l in Path('cert/RAD/memlog_v1.jsonl').read_text().splitlines() if l.strip()]
phase = lambda r: ('port' if r['lanes']['engineer']['rss_gib'] > 0.5 else '') + ('+stage2' if r['lanes']['physicist']['rss_gib'] > 1.0 else '') \
    + ('+release' if r['lanes']['release']['rss_gib'] > 0.5 else '')
groups = {}
for r in rows:
    groups.setdefault(phase(r).strip('+') or 'idle', []).append(r)
swap0 = rows[0]['swap_mib']
summary = {k: dict(minutes=len(v), min_free=min(r['mp_free'] for r in v), max_swap_rise=round(max(r['swap_mib'] for r in v) - swap0, 1),
                   min_free_inactive=min(r['free_inactive_gib'] for r in v), workers=sorted({sum(r['rad_workers'].values()) for r in v}))
           for k, v in groups.items()}
rel = groups.get('release', [])
ok = len(rel) >= 10 and min(r['mp_free'] for r in rel) >= 60 and max(r['swap_mib'] for r in rel) - rel[0]['swap_mib'] <= 64      # swap flat WITHIN the release-only minutes
rule = 'ready_table' if ok else 'release'
p = Path('cert/RAD/HOLD_RULE')
old = p.read_text().strip() if p.exists() else 'release'
stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
if rule != old:
    p.rename(p.with_name('HOLD_RULE.archived_%s_%s' % (stamp.replace(':', '').replace('-', ''), old)))
    p.write_text(rule + '\n')
line = ('[trainer] %s HOLD RULE DECIDED by the pre-declared rule (cert/RAD/hold_decide_v1.py on %d samples of cert/RAD/memlog_v1.jsonl, %s .. %s): '
        'HOLD_RULE = %s%s. Per phase (minutes, min memory_pressure free %%, max swap rise MiB, min free+inactive GiB, our worker counts): %s'
        % (stamp, len(rows), rows[0]['utc'], rows[-1]['utc'], rule, ' (changed from %s)' % old if rule != old else ' (unchanged)',
           '; '.join('%s %d min, free >= %d%%, swap +%s, f+i >= %.1f, workers %s' % (k, v['minutes'], v['min_free'], v['max_swap_rise'],
                                                                                v['min_free_inactive'], v['workers']) for k, v in sorted(summary.items()))))
with open('../TEAM_LOG.md', 'a') as f:
    f.write(line + '\n')
print(line)
