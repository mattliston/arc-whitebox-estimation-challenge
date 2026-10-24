#!/usr/bin/env python3
"""TAG opus-physicist T115 (lead 10-11 19:0xZ, option B): the trainer's generated-seed worker v4 (unchanged, through runpy) on the PINNED otr_m.py copy with the
engineer's star drop-in v2 aliased (ss -1), for the PLAIN torch card (requests without 'cutsim': the card's own QR join, not the native Nystrom join), with the
old tier's JOIN SCHEDULE of this worker process taken from the environment: T115_JOINS = '<layer>:<cohort>,<cohort>,.../...' replaces s2card.EVERY3
(fired: 7:0,1,2/10:3,4,5/13:6,7,8; the request's 'sched' gives one rank per join). Per network (tag T115_NAME_<seed>) the push's per-layer column counts
(all columns, old-tier columns) go to <run>/push_counts.jsonl for the engineer's bill price. Arithmetic otherwise = worker_generated_v4.
usage: T115_NAME=early T115_JOINS='6:0,1,2/9:3,4,5/12:6,7,8' python -B ../physicist/t115_worker_joins.py <run dir> <worker index>   (from O/trainer)"""
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER, FROZEN_M = HERE.parent / 'trainer', HERE / 'otr_frozen_bba0dea0'
RUN = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TRAINER))
sys.path.insert(0, str(FROZEN_M))
import otr_m as _M_FROZEN
assert Path(_M_FROZEN.__file__).resolve().parent == FROZEN_M and hashlib.sha256(Path(_M_FROZEN.__file__).read_bytes()).hexdigest().startswith('bba0dea0'), _M_FROZEN.__file__
sys.path.append(str(HERE.parent / 'engineer' / 'a33_starsum'))
import star_ss_torch_v2
sys.modules['star_ss_torch'] = star_ss_torch_v2
import otr_engine as E
import s2card
JOINS = {int(g.split(':')[0]): [int(c) for c in g.split(':')[1].split(',')] for g in os.environ['T115_JOINS'].split('/')}
s2card.EVERY3.clear(); s2card.EVERY3.update(JOINS)
NAME, CUR = os.environ.get('T115_NAME', 'joins'), {}
print('T115 worker: joins', s2card.EVERY3, 'name', NAME, flush=True)
run0 = E.run_network


def run_network(rank, *args, **kwargs):
    CUR['rank'] = int(rank); CUR['counts'] = {}
    rec = run0(rank, *args, **kwargs)
    with (RUN / 'push_counts.jsonl').open('a') as fh:
        fh.write(json.dumps(dict(tag='%s_%d' % (NAME, CUR['rank']), joins={str(k): v for k, v in s2card.EVERY3.items()}, counts=CUR['counts'])) + '\n')
    return rec
E.run_network = run_network
install0 = E.M.install


def install(cd):
    out = install0(cd)
    push0 = cd.shared['push']

    def push(tt, layer, *a, **k):
        r = push0(tt, layer, *a, **k)
        CUR.setdefault('counts', {})[str(layer)] = dict(cols=int(r[0].shape[1]), old=int(r[4]))
        return r
    cd.shared['push'] = push
    return out
E.M.install = install
sys.argv = [str(TRAINER / 'worker_generated_v4.py')] + sys.argv[1:]
runpy.run_path(str(TRAINER / 'worker_generated_v4.py'), run_name='__main__')
