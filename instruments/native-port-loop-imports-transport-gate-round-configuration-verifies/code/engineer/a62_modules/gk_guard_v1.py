"""TAG opus_engineer2 a62 gk_guard_v1 (file written 2026-10-12T22:25:01Z; lead 21:5xZ MANDATORY GUARD: from the absorb round on, the stack builder skips the
gk11 card delta, so native files carry gk ONLY through the port; a silent loss costs about +0.3%): the port loop's gk assertion for round N.
  - absorb round M = the earliest of ABSORB.json's from_round for the delta named gk11 and the rounds of anchored TEAM_LOG lines
    '[trainer] <stamp> ABSORBED gk11 at S<k>' (the stack builder's own skip rule, stack_deltas_v2.py); none, or N < M: 'GK GUARD n/a', exit 0
  - N >= M: the expected strengths are the pbase gsk / gk4 / gbk of round N's effective ABSORB carrier list (cert/RAD/absorb_v1.py's merge: every
    delta with from_round <= N, later ones override), or gsk = gk4 = 1.0, gbk = 0.0 (lead 21:5xZ) when that list has no non-zero gsk / gk4;
    (1) the round's config.json patch.ncar must have a pbase entry with exactly these gsk / gk4 / gbk (gbk absent = 0);
    (2) every file given must carry the gk edit (def _gq + the six edits' marker lines) and its LAST module-level _GSK / _GK4 / _GBK
        assignments must equal them;
    -> 'GK GUARD OK ...' exit 0, else 'GK GUARD FAIL <reason>' exit 4.
usage: gk_guard_v1.py <round N> <config.json> [<candidate.py> ...]   (env ABSORB_JSON / TEAM_LOG override the two sources, for tests)"""
import ast
import json
import os
import re
import sys
from pathlib import Path

O = Path(__file__).resolve().parents[2]
N, cfg_path, files = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
absorb_path = Path(os.environ.get('ABSORB_JSON', O / 'trainer/cert/RAD/ABSORB.json'))
log_path = Path(os.environ.get('TEAM_LOG', O / 'TEAM_LOG.md'))
MARKERS = ('def _gq(', 'gq = _gq(g, d, aux, bpre)', 'gq * bpre', '_ps_g2 = gq ** 2', '_e2_H(gq.astype(fnp.float32)) * alive',
           '_E2_LAYER.update(ga_gate=gq, ga_alive=alive, ga=_e2_ga)', '_f4c_step(l, right if len(lam) else None, gq, d, live_indices')


def fail(why):
    print('GK GUARD FAIL S%d: %s' % (N, why))
    sys.exit(4)


deltas = json.loads(absorb_path.read_text())['deltas'] if absorb_path.is_file() else []
rounds = [int(d['from_round']) for d in deltas if d.get('name') == 'gk11']
rounds += [int(m.group(1)) for m in re.finditer(r'^\[trainer\] \S+ ABSORBED gk11 at S(\d+)', log_path.read_text(), re.M)]
if not rounds or N < min(rounds):
    print('GK GUARD n/a (S%d; gk11 absorbed from %s)' % (N, 'S%d' % min(rounds) if rounds else 'no round'))
    sys.exit(0)
M = min(rounds)
patch = {}
for d in deltas:
    if int(d['from_round']) <= N:
        patch.update(d['patch'])
pb = [e for e in patch.get('ncar', []) if e.get('module') == 'pbase']
want = tuple(float(pb[-1].get(k, 0.0)) for k in ('gsk', 'gk4', 'gbk')) if pb else (0.0, 0.0, 0.0)
if not (want[0] or want[1]):
    want = (1.0, 1.0, 0.0)
cfg = json.loads(cfg_path.read_text())
got = [e for e in cfg.get('patch', {}).get('ncar', []) if e.get('module') == 'pbase']
if not got:
    fail('config %s has no pbase entry in patch.ncar (gk11 absorbed from S%d): the round would port WITHOUT gk' % (cfg_path, M))
have = tuple(float(got[-1].get(k, 0.0)) for k in ('gsk', 'gk4', 'gbk'))
if have != want:
    fail('config pbase gsk / gk4 / gbk %r, expected %r (gk11 absorbed from S%d)' % (have, want, M))
seen = []
for f in files:
    text = Path(f).read_text()
    miss = [m for m in MARKERS if m not in text]
    if miss:
        fail('%s lacks the gk edit (%d of %d markers missing, first: %r)' % (f, len(miss), len(MARKERS), miss[0]))
    vals = []
    for name in ('_GSK', '_GK4', '_GBK'):
        found = re.findall(r'^%s = ([^#\n]+)' % name, text, re.M)
        if not found:
            fail('%s has no module-level %s' % (f, name))
        vals.append(float(ast.literal_eval(found[-1].strip())))
    if tuple(vals) != want:
        fail('%s carries _GSK / _GK4 / _GBK %r, expected %r' % (f, tuple(vals), want))
    seen.append(os.path.basename(os.path.dirname(f)) or f)
print('GK GUARD OK S%d (gk11 from S%d): config pbase gsk %r gk4 %r gbk %r; files carrying exactly these: %s' % (N, M, *want, ', '.join(seen) or 'none checked'))
