"""TAG opus_engineer a62 (file written 2026-10-12T17:31Z; release 17:2xZ fastgate_v1 trigger + the lead's mandatory harness checks): checks a
stack twin's sessions against release/fastgate_pin_v1.json and prints the trigger line body
    FASTGATE <tag> READY: twin <path> <sha8>, base <quick json>, normal <dir>, slow <dir>
or, on any miss, 'NOT READY: <reasons>' (exit 1). Checks: the harness files' sha256 = the pin (boundary_run, client, setup clock, server,
rusage); each session's manifest.json and launcher.json carry the twin's full sha256; ranks [35, 100]; both short setup (the emulated
clock), normal at slowdown 1.0, slow at 1.3; sites and profiled off; client flopscope = the pin's; client and server exit 0; guard record completed with exit
0; rows for 35 and 100; the base json's _pred_<rank>.npy next to it for every rank it lists.
usage: fastgate_post_v1.py <tag> <twin candidate.py> <base quick_keys json> <normal session dir> <slow session dir>"""
import hashlib
import json
import sys
from pathlib import Path

tag, twin, base, normal, slow = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), Path(sys.argv[4]).resolve(), Path(sys.argv[5]).resolve()
O = Path(__file__).resolve().parents[2]
pin = json.load(open(O/'release/fastgate_pin_v1.json'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
full = sha(twin)
bad = []
for name, want in pin['harness'].items():
    if sha(O/'release'/name) != want:
        bad.append('harness %s changed since the pin' % name)
for kind, d, short, slowdown in (('normal', normal, True, 1.0), ('slow', slow, True, 1.3)):   # boundary_run's slow session is the short setup x 1.3
    try:
        m, la, g = (json.load(open(d/f)) for f in ('manifest.json', 'launcher.json', 'guard.json'))
        ranks = sorted(json.loads(line)['rank'] for line in open(d/'rows.jsonl') if '"rank"' in line)
    except Exception as e:
        bad.append('%s session unreadable (%s)' % (kind, e))
        continue
    checks = [(m.get('sha256') == full, 'manifest sha'), (la.get('sha256') == full, 'launcher sha'), (m.get('ranks') == [35, 100], 'ranks %s' % m.get('ranks')),
              (bool(m.get('short_setup')) == short, 'short_setup %s' % m.get('short_setup')), (float(m.get('setup_slowdown', 0)) == slowdown, 'setup_slowdown %s' % m.get('setup_slowdown')),
              (m.get('sites') is False and m.get('profiled') is False, 'sites / profiled on'), (m.get('client_flopscope') == pin['client_flopscope'], 'flopscope %s' % m.get('client_flopscope')),
              (la.get('client_exit') == 0 and la.get('server_exit') == 0, 'exits %s / %s' % (la.get('client_exit'), la.get('server_exit'))),
              (g.get('completed') is True and g.get('exit_code') == 0 and g.get('error') is None, 'guard not completed cleanly'), (ranks == [35, 100], 'rows %s' % ranks)]
    bad += ['%s: %s' % (kind, what) for ok, what in checks if not ok]
try:
    ranks = [r['rank'] for r in json.load(open(base))['rows']]
    missing = [r for r in ranks if not (base.parent/('%s_pred_%d.npy' % (base.stem, r))).exists()]
    if not ranks or missing:
        bad.append('base: no rows or _pred files missing for %s' % missing)
except Exception as e:
    bad.append('base unreadable (%s)' % e)
if bad:
    print('NOT READY: ' + '; '.join(bad))
    sys.exit(1)
print('FASTGATE %s READY: twin %s %s, base %s, normal %s, slow %s' % (tag, twin, full[:8], base, normal, slow))
