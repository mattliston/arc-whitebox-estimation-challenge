"""TAG opus-trainer RAD (written 2026-10-12T18:5xZ; lead 18:3xZ stack protocol): ABSORB a stacked card delta into the RAD line's start.
Data file cert/RAD/ABSORB.json (absent = nothing absorbed, every make file behaves exactly as before):
  {"deltas": [{"name": "gk", "from_round": 53, "patch": {"ncar": [... the FULL list ...]}, "rule_c": "loo<0" | "all" | null, "what": "..."}]}
  - every round N >= from_round merges the delta's patch into its config patch at the config stage (later deltas override a key, so an 'ncar'
    patch must be the full carrier list including every earlier absorbed change);
  - the round N == from_round posts '[trainer] <stamp> ABSORBED <name> at S<N> ...' after its config is written, and with rule_c set runs
    radlab's rule (c) ONCE: the strength arms of the latest earlier round that sized any (S<k>/arms_fixed_v1 names f<fam>_<lo>_<hi>, never
    'fagg'), at that round's trial sizes, without kappa and without arms with an anchored engineer 'S<k> PORT PARITY <arm>: FAILED' line;
    'loo<0' keeps those with single-arm LOO < 0 in S<k>'s solve (radlab's rule as written), 'all' keeps every one."""
import hashlib
import json
import os
import re
from pathlib import Path

read = lambda p: json.loads(Path(p).read_text())
utc = lambda: os.popen('date -u +%FT%TZ').read().strip()
ABSORB = Path('cert/RAD/ABSORB.json')


def strength_name_ok(a):      # a strength (fixed forward) arm name f<family>_<lo>_<hi>; 'fagg' (rule (d)) is not one
    return re.fullmatch(r'f[A-Za-z0-9]+_[0-9]+_[0-9]+', a) is not None


def absorb(cfg, n, path=ABSORB):
    if not Path(path).exists():
        return cfg, []
    deltas = [d for d in read(path)['deltas'] if int(d['from_round']) <= n]
    if not deltas:
        return cfg, []
    for d in deltas:
        cfg['patch'].update(d['patch'])
    now = [d['name'] for d in deltas if int(d['from_round']) == n]
    rc = [d['rule_c'] for d in deltas if int(d['from_round']) == n and d.get('rule_c')]
    cfg['absorbed'] = [d['name'] for d in deltas]
    cfg['absorb_file'] = dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    if rc:
        cfg['rule_c_once'] = 'all' if 'all' in rc else rc[0]
    return cfg, now


def post_absorbed(cfg, now, n, path=ABSORB):
    if not now:
        return
    what = '; '.join('%s: %s' % (d['name'], d.get('what', '')) for d in read(path)['deltas'] if d['name'] in now)
    with open('../TEAM_LOG.md', 'a') as log:
        log.write('[trainer] %s ABSORBED %s at S%d (cert/RAD/ABSORB.json sha %s): %s. From S%d on every RAD request carries it through the config '
                  'patch (keys %s); rule (c) once in S%d: %s.\n' % (utc(), ' + '.join(now), n, cfg['absorb_file']['sha256'][:12], what, n,
                                                                     ', '.join(sorted(k for d in read(path)['deltas'] if d['name'] in now for k in d['patch'])),
                                                                     n, cfg.get('rule_c_once') or 'no'))


def rule_c_once(n, rule):
    eng = Path('../TEAM_LOG.md').read_text().splitlines()
    for k in range(n - 1, 49, -1):
        fx, sv = Path('cert/RAD/S%d/arms_fixed_v1/arms.json' % k), Path('cert/RAD/S%d/solve_v1.json' % k)
        if not (fx.exists() and sv.exists()):
            continue
        sizes = {a: v for a, v in read(fx)['size'].items() if strength_name_ok(a)}
        if not sizes:
            continue
        failed = {m.group(1) for m in (re.match(r'\[engineer\] \S+ S%d PORT PARITY ([A-Za-z0-9_]+): FAILED' % k, l) for l in eng) if m}
        loo = read(sv)['arm_alone_loo_pct']
        strength = [a for a in sizes if not a.startswith('fkappa') and a not in failed and (rule == 'all' or loo.get(a, 0.0) < 0)]
        spec = ['%s:%s:%s:%s' % (*a[1:].rsplit('_', 2), sizes[a]) for a in strength]
        return 'S%d' % k, strength, spec
    return None, [], []
