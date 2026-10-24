"""Target-free old common-u read capture; stock NumPy-free client only.

Binary fetches are diagnostic work inside the prediction context. They cannot
be cited as bill/runtime qualification. The estimator source and return values
are untouched, and the observer retains no native array handles.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE_SHA = '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


class Observer:
    def __init__(self, mod, outdir, recipe):
        assert hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() == SOURCE_SHA
        assert mod._FAST_RESCUE_FRAC == {}
        self.mod, self.recipe = mod, recipe
        self.out = Path(outdir) / 'old_su_capture_v1'
        self.out.mkdir(exist_ok=False)
        self.rank = self.phase = None
        self.finished = False
        self.files, self.counts, self.pending = [], {}, {}
        self.total_bytes = 0
        self.parent_profile = sys.getprofile()
        self.code = mod._ubx_product.__code__
        assert self.code.co_firstlineno == 14043
        write_json(self.out / 'METADATA_v1.json', dict(
            utc=utc(), source_sha256=SOURCE_SHA,
            hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            target_free=True, reference_submission=337422,
            allowed_ranks=[35, 150], layers=list(range(10, 15)),
            raw_binary_dtype='float32 except live_indices integer',
            byte_limit=1500000000,
            limitation='Observation only; fetched data invalidate diagnostic bill/runtime; no truth files in plan.'))
        sys.setprofile(self.profile)

    def begin(self, rank, phase):
        assert not self.finished and not self.pending
        assert int(rank) in (35, 150) and phase == 'scored'
        self.rank, self.phase = int(rank), str(phase)

    def fetch(self, value, stem):
        data, shape, dtype = value._fetch_data()
        assert isinstance(data, bytes), type(data)
        self.total_bytes += len(data)
        assert self.total_bytes <= 1500000000, 'diagnostic byte limit'
        path = self.out / (stem + '.bin')
        with path.open('xb') as f:
            f.write(data)
        rec = dict(path=path.name, bytes=len(data), shape=list(shape), dtype=str(dtype),
                   sha256=hashlib.sha256(data).hexdigest())
        del data
        return rec

    @staticmethod
    def contexts(frame):
        state = live = None
        caller = frame.f_back
        while caller is not None:
            scope = caller.f_locals
            possible = scope.get('state')
            if state is None and isinstance(possible, dict) and 'pre_basis' in possible and 'columns' in possible:
                state = possible
            if live is None and 'live_indices' in scope:
                live = scope['live_indices']
            caller = caller.f_back
        assert state is not None and state['pre_basis'] is not None
        assert live is not None
        return state, live

    def profile(self, frame, event, arg):
        if self.parent_profile is not None:
            self.parent_profile(frame, event, arg)
        if self.finished or self.rank is None or frame.f_code is not self.code:
            return
        scope = frame.f_locals
        if scope.get('leg') != 'ut':
            return
        layer = int(self.mod._UVP_CUR[0])
        if not 10 <= layer <= 14:
            return
        if event == 'call':
            state, live = self.contexts(frame)
            uv = scope['uv']
            group = state['lazy']
            part = 'materialized' if group is None else ('base' if uv is state['uvp'][1] else 'join')
            key = (self.rank, layer, part)
            assert key not in self.counts, ('duplicate', key)
            self.counts[key] = 1
            stem = 'rank%04d_L%02d_%s' % key
            record = dict(rank=self.rank, phase=self.phase, layer=layer, part=part,
                          pair_blocks=[list(x) for x in uv[2]], common_rank=int(uv[3]),
                          old_columns=int(state['columns']), physical_rank=int(state['rank']),
                          Su=self.fetch(scope['left'], stem + '_Su'),
                          FuT=self.fetch(scope['right'], stem + '_FuT'),
                          pre_basis=self.fetch(state['pre_basis'], stem + '_pre_basis'),
                          live_indices=self.fetch(live, stem + '_live_indices'))
            if group is not None and part == 'base':
                record['lazy_map'] = self.fetch(group['map'], stem + '_lazy_map')
            self.pending[id(frame)] = (stem, record)
        elif event == 'return':
            stem, record = self.pending.pop(id(frame))
            assert arg is not None, 'native product failed; incomplete capture'
            record['native_inner'] = self.fetch(arg, stem + '_native_inner')
            path = self.out / (stem + '.json')
            write_json(path, record)
            self.files.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def finish(self):
        assert not self.finished and not self.pending
        self.finished = True
        sys.setprofile(self.parent_profile)
        ranks = sorted({key[0] for key in self.counts})
        for rank in ranks:
            assert {(l, p) for r, l, p in self.counts if r == rank} == {(10, 'materialized'), (11, 'materialized'), (12, 'materialized'), (13, 'base'), (13, 'join'), (14, 'base'), (14, 'join')}
        result = dict(utc=utc(), complete=True, ranks=ranks, files=self.files,
                      counts=[list(key) + [value] for key, value in sorted(self.counts.items())],
                      source_sha256=SOURCE_SHA, total_bytes=self.total_bytes,
                      target_free=True, native_handles_retained=False)
        write_json(self.out / 'CAPTURE_COMPLETE_v1.json', result)
        return result


def install(mod, outdir, recipe):
    return Observer(mod, outdir, recipe)
