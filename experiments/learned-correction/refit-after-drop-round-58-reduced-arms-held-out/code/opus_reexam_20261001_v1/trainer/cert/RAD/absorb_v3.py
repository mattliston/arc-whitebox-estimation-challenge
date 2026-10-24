"""TAG opus-trainer RAD (written 2026-10-13T16:1xZ; lead 15:5xZ: if T146 passes and the c0 literal fires, S57's start must carry the fired c0): = absorb_v2.py
+ a third kind of delta in cert/RAD/ABSORB.json, 'start_pl_add': {family: 16 INCREMENTS} for a family every point already stores (c0 = dial + shift,
stored -0.0122 on all layers), which absorb_v2's add-if-absent rule would never touch. The increments go into <round>/start_v1/params.pt when the
round's start point PREDATES the delta (the frozen point of a round < from_round): a later round continuing from an absorbing round's point keeps its own
(RAD-sized) values, and a round that restarts from an older point gets the increments again. 'start_pl' (absolute, families the start lacks) and
'patch' deltas, the ABSORBED lines and rule (c) once are absorb_v2's / absorb_v1's. With no 'start_pl_add' delta it behaves exactly as absorb_v2.
usage: from absorb_v3 import absorb, post_absorbed (cert/RAD on sys.path); absorb(cfg, n, home=HOME)."""
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

from absorb_v1 import ABSORB, read, rule_c_once, strength_name_ok      # noqa: F401 (re-exported)
from absorb_v1 import post_absorbed as post_absorbed_v1
from absorb_v1 import absorb as absorb_v1


def source_round(params):
    m = re.search(r'/S(\d+)/(?:point|start)_v1/params(?:_exportable_v1)?\.pt$', str(params))
    assert m, 'cannot tell which round the start point comes from: %s' % params
    return int(m.group(1))


def absorb(cfg, n, path=ABSORB, home=None):
    cfg, now = absorb_v1(cfg, n, path)
    if not Path(path).exists():
        return cfg, now
    deltas = [d for d in read(path)['deltas'] if int(d['from_round']) <= n]
    start_pl, incs = {}, {}
    for d in deltas:
        start_pl.update(d.get('start_pl') or {})
    if any(d.get('start_pl_add') for d in deltas):
        src = source_round(cfg['params'])
        for d in deltas:
            if d.get('start_pl_add') and src < int(d['from_round']):
                for f, v in d['start_pl_add'].items():
                    incs[f] = [a + float(b) for a, b in zip(incs.get(f, [0.0] * 16), v)]
    if not start_pl and not incs:
        return cfg, now
    import torch
    if '.' not in sys.path:
        sys.path.insert(0, '.')      # the make file runs from O/trainer (its chain's cwd), where otr_common.py lives
    from otr_common import load_params, load_pl, save_params
    arrays, dials = load_params(torch, cfg['params']); pl = {k: [float(x) for x in v] for k, v in load_pl(torch, cfg['params']).items()}
    add = {f: [float(x) for x in v] for f, v in start_pl.items() if f not in pl}
    missing = sorted(f for f in incs if f not in pl)
    assert not missing, 'start_pl_add needs families the start point stores: %s' % missing
    if not add and not incs:
        cfg['start_pl'] = dict(applied={}, kept_from_start=sorted(start_pl), source=cfg['params'])
        return cfg, now
    assert home is not None and all(len(v) == 16 for v in list(add.values()) + list(incs.values())), 'start_pl needs the round home and 16 values per family'
    out = Path(home) / 'start_v1'
    out.mkdir()
    pl.update(add)
    for f, v in incs.items():
        pl[f] = [a + b for a, b in zip(pl[f], v)]
    save_params(torch, out / 'params.pt', arrays, dials, pl)
    shutil.copy2(cfg['hu'], out / 'hu.json')
    cfg['start_pl'] = dict(applied=add, kept_from_start=sorted(set(start_pl) - set(add)), increments=incs, source=cfg['params'], source_hu=cfg['hu'],
                           sha256=hashlib.sha256((out / 'params.pt').read_bytes()).hexdigest())
    cfg['params'], cfg['hu'] = str((out / 'params.pt').resolve()), str((out / 'hu.json').resolve())
    return cfg, now


def post_absorbed(cfg, now, n, path=ABSORB):
    """absorb_v2's lines (patch deltas: absorb_v1's; start_pl deltas: one line each), and one line per start_pl_add delta, same anchored head
    '[trainer] <stamp> ABSORBED <id> at S<N>'."""
    if not now:
        return
    deltas = {d['name']: d for d in read(path)['deltas'] if d['name'] in now}
    plain = [x for x in now if not deltas[x].get('start_pl') and not deltas[x].get('start_pl_add')]
    if plain:
        post_absorbed_v1(cfg, plain, n, path)
    sp = cfg.get('start_pl', {})
    for name in now:
        d = deltas[name]
        if not (d.get('start_pl') or d.get('start_pl_add')):
            continue
        stamp = os.popen('date -u +%FT%TZ').read().strip()
        fams = sorted(set(d.get('start_pl') or {}) | set(d.get('start_pl_add') or {}))
        how = '; '.join((['absolute values for the families it lacks; a start point that carries them keeps its own'] if d.get('start_pl') else []) +
                        (['increments on the stored values when the start point predates S%d; a point from S%d on keeps its own' % (n, n)] if d.get('start_pl_add') else []))
        with open('../TEAM_LOG.md', 'a') as log:
            log.write('[trainer] %s ABSORBED %s at S%d (cert/RAD/ABSORB.json sha %s): %s. From S%d on the round START point carries %s (%s; written into '
                      'cert/RAD/S%d/start_v1/params.pt from %s), so base, arms, freeze and check run that card and the frozen point hands it to the port '
                      '(read_from_pt).\n'
                      % (stamp, name, n, cfg['absorb_file']['sha256'][:12], d.get('what', ''), n, ', '.join(fams), how, n,
                         '/'.join(Path(sp.get('source', '?')).parts[-3:]) if (sp.get('applied') or sp.get('increments')) else 'the start point itself (already carried)'))
