"""TAG opus-trainer RAD (written 2026-10-12T18:3xZ; lead 18:1xZ, after S51's escal): the ZERO-ARM GUARD of the make files' arm stage (S52+).
An arm whose parameter file equals the round's start point bit for bit (all 28 arrays, the dials, the per-layer shifts with all-zero families
ignored, and the head json) moves nothing; sizing it wastes 128 evals and its column is pure eval noise (S51 escal: weight 16.54 on rounding).
drop_zero_arms(dirs, cfg, name) removes such arms from each arms dir's arms.json 'live' list BEFORE any arm is queued (the original arms.json is
kept as arms.json.archived_<stamp>_zero), writes one TEAM_LOG line when it drops any, and returns the dropped names.
Usage (python, from O/trainer): from zero_arms_v1 import drop_zero_arms (cert/RAD on sys.path); python -B cert/RAD/zero_arms_v1.py <cfg json> <dir>[,<dir>] = dry run."""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '.')
from otr_common import load_params, load_pl

read = lambda p: json.loads(Path(p).read_text())
utc = lambda: os.popen('date -u +%FT%TZ').read().strip()


def _nonzero_pl(pl):
    return {k: [float(x) for x in v] for k, v in pl.items() if any(float(x) != 0.0 for x in v)}


def zero_arms(dirs, cfg):
    import torch
    a0, d0 = load_params(torch, cfg['params']); p0 = _nonzero_pl(load_pl(torch, cfg['params'])); h0 = read(cfg['hu'])
    out = {}
    for d in map(Path, dirs):
        for a in read(d / 'arms.json')['live']:
            f = d / ('arm_%s.pt' % a)
            a1, d1 = load_params(torch, f)
            if (d1 == d0 and set(a1) == set(a0) and all(torch.equal(a1[k], a0[k]) for k in a0)
                    and _nonzero_pl(load_pl(torch, f)) == p0 and read(d / ('arm_%s_hu.json' % a)) == h0):
                out.setdefault(str(d), []).append(a)
    return out


def drop_zero_arms(dirs, cfg, name):
    found = zero_arms(dirs, cfg)
    stamp = utc()
    for d, arms in found.items():
        j = Path(d) / 'arms.json'
        shutil.copy2(j, Path(d) / ('arms.json.archived_%s_zero' % stamp.replace('-', '').replace(':', '')))
        x = read(j)
        x['live'] = [a for a in x['live'] if a not in arms]
        x['zero_arms_dropped'] = arms
        j.write_text(json.dumps(x, indent=1))
    dropped = [a for arms in found.values() for a in arms]
    if dropped:
        with open('../TEAM_LOG.md', 'a') as log:
            log.write('[trainer] %s %s ZERO ARMS DROPPED before sizing (zero-arm guard, lead 18:1xZ): %s; each one\'s parameter file equals the start '
                      'point bit for bit (28 arrays, dials, per-layer shifts, head), so it would size nothing but eval noise. Original arms.json kept '
                      'as arms.json.archived_*_zero.\n' % (stamp, name, ' '.join(dropped)))
    return dropped


if __name__ == '__main__':      # dry run: report, change nothing
    print('zero arms:', zero_arms(sys.argv[2].split(','), read(sys.argv[1])))
