#!/usr/bin/env python3
"""TAG opus-physicist T103: the trainer's generated-seed worker (trainer/worker_generated_v4.py, unchanged, run through runpy) plus per-band FLOORS
of the card's young-bank selections for the module audit's bill side (lead 10-11 RESTART: is our 55% of bill in the young bank well spent?).
A request whose cutsim opts carry 't103': {'births': [lo, hi, floor], 'resp': [lo, hi, floor]} (either key optional; removed before the pinned
simulator sees its options) raises, at layers lo..hi, the floor of the read's response rows (min(g, 1 - g) >= floor among live units) and / or of
the births (min(g, 1 - g) >= floor) from the card's 0.01: the score handed to the card's select is scaled by 0.01 / floor (order kept; the card
rounds the count to 32 as before). Every request's per-layer counts (live, response rows, births) go to <run>/sel_counts.jsonl with its tag.
Without 't103' the arithmetic is worker_generated_v4 exactly.
v2 (15:35Z): runs on the pinned otr_m.py copy in physicist/otr_frozen_bba0dea0.
usage: as worker_generated_v4.py, run from O/trainer: python -B ../physicist/t103_worker_floors.py <run dir> <worker index>"""
import json
import runpy
import sys
from pathlib import Path

TRAINER = Path(__file__).resolve().parent.parent / 'trainer'
RUN = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TRAINER))
FROZEN_M = Path(__file__).resolve().parent / 'otr_frozen_bba0dea0'      # v2: the pinned otr_m.py (sha bba0dea0 = the v11 certificate's), first on the
sys.path.insert(0, str(FROZEN_M))                                        # path: the trainer edited his live copy at 15:30Z (edge inputs)
import hashlib
import otr_m as _M_FROZEN
assert Path(_M_FROZEN.__file__).resolve().parent == FROZEN_M and hashlib.sha256(Path(_M_FROZEN.__file__).read_bytes()).hexdigest().startswith('bba0dea0'), _M_FROZEN.__file__
import otr_cut_runtime_v2 as RT

enable0, CUR = RT.enable, {}


def enable(engine):
    configure0 = enable0(engine)
    install_runtime = engine.M.install

    def configure(spec):
        CUR.clear()
        if spec and ('t103' in (spec.get('opts') or {}) or 't103tag' in (spec.get('opts') or {})):
            opts = dict(spec['opts'])
            CUR['arm'] = opts.pop('t103', {}); CUR['tag'] = opts.pop('t103tag', None)
            spec = dict(spec, opts=opts)
        configure0(spec)

    def install(cd):
        out = install_runtime(cd)
        sel0 = cd.select
        slot, counts = {}, {}

        def select(score):
            l = cd.call_layer; k = slot.get(l, 0); slot[l] = k + 1
            role = {0: 'alive', 1: 'resp', 2: 'births'}.get(k)
            band = CUR.get('arm', {}).get(role) if role in ('resp', 'births') else None
            if band and band[0] <= l <= band[1] and not getattr(cd, 'm_side', {}).get('again'):
                score = score * (0.01 / float(band[2]))
            r = sel0(score)
            if role and not getattr(cd, 'm_side', {}).get('again'):
                counts.setdefault(l, {})[role] = int(r[0].numel())
                if CUR.get('tag') and role == 'births' and l == 14:
                    with (RUN / 'sel_counts.jsonl').open('a') as fh:
                        fh.write(json.dumps(dict(tag=CUR['tag'], counts={str(k_): v for k_, v in sorted(counts.items())})) + '\n')
            return r
        cd.select = select
        return out
    engine.M.install = install
    return configure


RT.enable = enable
sys.argv = [str(TRAINER / 'worker_generated_v4.py')] + sys.argv[1:]
runpy.run_path(str(TRAINER / 'worker_generated_v4.py'), run_name='__main__')
