#!/usr/bin/env python3
"""TAG opus-physicist T129 (lead 10-12 16:2xZ, priority sprint test: RAISE the births floor at late layers) = t103_worker_floors_v3.py's floor patch on the
trainer's CURRENT worker: worker_generated_v12.py (otr_m_v5, the S48 / S50 card: read column cut, star strength, k21n / rsy families), run UNCHANGED through
runpy after asserting its and otr_m_v5's shas. A request whose cutsim opts carry 't103': {'births': [lo, hi, floor]} (and 't103tag') scales the births
selection's score by 0.01 / floor at layers lo..hi (floor > 0.01 = FEWER births; the live and response rosters untouched: births only); per-layer counts
(live, response rows, births) to <run>/sel_counts.jsonl and the push's per-layer columns (all, old tier) to <run>/push_counts.jsonl, both with the tag.
Without 't103' the arithmetic is worker_generated_v12 exactly.
usage (from O/trainer): python -B ../physicist/t129_worker_floors_v12.py <run dir> <worker index>"""
import json
import runpy
import sys
from pathlib import Path

TRAINER = Path(__file__).resolve().parent.parent / 'trainer'
RUN = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TRAINER))
import hashlib
for _name, _sha in (('worker_generated_v12.py', 'a88ae6ac5f37f16c18accbbbb2b873863e677ebff88a9ea2e4abc5d34d4851d6'),
                    ('otr_m_v5.py', '7553b39e407eada61c3a2b4047e9af340c032f918defc07b770c3bf1d5aefdc2')):
    assert hashlib.sha256((TRAINER / _name).read_bytes()).hexdigest() == _sha, _name
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
        push0 = cd.shared['push']
        pc = {}

        def push(tt, layer, *a, **k):
            r = push0(tt, layer, *a, **k)
            pc[str(layer)] = dict(cols=int(r[0].shape[1]), old=int(r[4]))
            if layer == 15 and CUR.get('tag'):
                with (RUN / 'push_counts.jsonl').open('a') as fh:
                    fh.write(json.dumps(dict(tag=CUR['tag'], counts=pc)) + '\n')
            return r
        cd.shared['push'] = push
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
sys.argv = [str(TRAINER / 'worker_generated_v12.py')] + sys.argv[1:]
runpy.run_path(str(TRAINER / 'worker_generated_v12.py'), run_name='__main__')
