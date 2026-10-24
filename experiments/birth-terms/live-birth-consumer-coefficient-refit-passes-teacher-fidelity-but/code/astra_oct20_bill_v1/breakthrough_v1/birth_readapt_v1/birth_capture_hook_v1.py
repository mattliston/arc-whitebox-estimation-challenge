"""Observe the exact original-CM16 birth consumer; no predictor mutation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE = '09fb70c86fa245d083f1db5d9c06dcde90cd71cbc8aaeaabb1bcb134aef59ded'
RANKS = (35, 150, 200, 250)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


class Observer:
    def __init__(self, mod, outdir, recipe):
        assert sha(mod.__file__) == SOURCE == recipe['teacher']['effective_sha256']
        assert mod._FAST_RESCUE_FRAC == {}
        assert tuple(mod._GNN_LITERAL['msg_w'][0]) == (48, 64)
        self.mod = mod
        self.recipe = recipe
        self.path = Path(outdir) / 'birth_consumer_v1'
        self.path.mkdir(exist_ok=False)
        self.parent = sys.getprofile()
        self.code = mod._e2_bdnh_coefficients.__code__
        assert self.code.co_name == '_e2_bdnh_coefficients' and self.code.co_firstlineno == 2375
        self.rank = None
        self.phase = None
        self.finished = False
        self.seen = set()
        self.files = []
        self.total_bytes = 0
        write(self.path / 'METADATA_v1.json', dict(utc=utc(), source_sha256=SOURCE,
            hook_sha256=sha(__file__), target_free=True, diagnostic_only=True,
            rows_per_layer=1024, layers=list(range(1, 15)),
            A_literal=mod._BDNH_A_LITERAL, operation='native birth readout observer',
            normalization='Actual native all1024-row centered/variance-standardized state',
            no_native_handles_retained=True, source_arithmetic_unchanged=True))
        sys.setprofile(self.profile)

    def begin(self, rank, phase):
        assert not self.finished and int(rank) in RANKS and phase == 'scored'
        self.rank = int(rank)
        self.phase = phase

    def fetch(self, value, stem, name, shape):
        assert tuple(value.shape) == tuple(shape)
        raw, actual_shape, dtype = value._fetch_data()
        assert list(actual_shape) == list(shape) and str(dtype) in ('float32', '<f4', '=f4', 'f4')
        data = bytes(raw)
        count = 1
        for n in shape:
            count *= n
        assert len(data) == count * 4
        self.total_bytes += len(data)
        assert self.total_bytes < 80_000_000
        file = self.path / (stem + '_' + name + '.bin')
        with file.open('xb') as f:
            f.write(data)
        return dict(path=file.name, shape=list(shape), dtype='<f4', bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest())

    def profile(self, frame, event, arg):
        if self.parent is not None:
            self.parent(frame, event, arg)
        if self.finished or self.rank is None or event != 'return' or frame.f_code is not self.code:
            return
        state = frame.f_locals
        layer = int(state['l'])
        assert 1 <= layer <= 14 and self.phase == 'scored'
        key = (self.rank, layer)
        assert key not in self.seen
        stem = 'rank%04d_L%02d' % key
        rec = dict(rank=self.rank, layer=layer, phase=self.phase,
                   source_sha256=SOURCE, target_free=True)
        for name, value, shape in (
            ('state', state['H'], (1024, 64)),
            ('standardized', state['standardized'], (1024, 64)),
            ('response', state['response'], (1024, 4)),
            ('coefficients', arg, (4, 1024)),
            ('cstar', state['cstar'], (4,)),
            ('onem', state['onem'], (4,)),
        ):
            rec[name] = self.fetch(value, stem, name, shape)
        path = self.path / (stem + '.json')
        write(path, rec)
        self.files.append(dict(path=path.name, sha256=sha(path), rank=self.rank, layer=layer))
        self.seen.add(key)

    def finish(self):
        assert not self.finished
        self.finished = True
        sys.setprofile(self.parent)
        ranks = sorted({r for r, _ in self.seen})
        assert ranks
        for rank in ranks:
            assert {l for r, l in self.seen if r == rank} == set(range(1, 15))
        result = dict(utc=utc(), complete=True, target_free=True, diagnostic_only=True,
                      source_sha256=SOURCE, ranks=ranks, files=self.files,
                      total_bytes=self.total_bytes, no_native_handles_retained=True,
                      original_profile_restored=True, source_arithmetic_unchanged=True)
        write(self.path / 'CAPTURE_COMPLETE_v1.json', result)
        return result


def install(mod, outdir, recipe):
    return Observer(mod, outdir, recipe)
