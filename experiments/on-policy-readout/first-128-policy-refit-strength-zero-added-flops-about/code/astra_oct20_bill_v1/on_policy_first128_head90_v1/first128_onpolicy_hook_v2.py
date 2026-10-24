"""Target-free source-specific first128/state capture; no estimator mutation.

Diagnostic fetches materialize native arrays and change request timing. Only
source-specific plain/capture anchors can establish prediction and bill parity.
"""
import gc
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n')

class Observer:
    def __init__(self, mod, outdir, recipe):
        self.mod = mod
        self.outdir = Path(outdir)/'gnn_full_state_v1'
        self.outdir.mkdir(exist_ok=False)
        self.recipe = dict(recipe)
        self.rank, self.phase = None, None
        self.seen, self.first_seen, self.pending = set(), set(), {}
        self.files, self.finished, self.skipped_layer0 = [], False, 0
        self.parent = sys.getprofile()
        self.code = mod._gop2_gnn_step.__code__
        self.first_code = mod._fh90_first_head.__code__
        self.gc_state = (gc.isenabled(), gc.get_threshold())
        self.alias_ids = {name: id(getattr(mod, name)) for name in
                          ('_gop2_gnn_step', '_fh90_first_head', '_hc90_hidden_head', '_E2Hold', '_hc_bias')}
        assert mod._GOP2_ON and mod._FAST_RESCUE_FRAC == {}
        assert sha(mod.__file__) == recipe['teacher']['sha256'] == recipe['teacher']['effective_sha256']
        assert recipe['target_free'] and not recipe['uses_network_truth']
        assert recipe['rows_per_layer'] == 1024 and recipe['layers'] == list(range(1, 16))
        assert recipe['source_family']['science'] == recipe['teacher']
        assert not set(recipe['source_family']['train'])&set(recipe['source_family']['validation'])
        self.metadata = dict(utc=utc(), source_sha256=sha(mod.__file__), hook_sha256=sha(__file__),
             rows_per_layer=1024, layers=list(range(1, 16)), target_free=True, reference_submission=recipe['reference_submission'],
             source_family=recipe['source_family'], source_manifest=recipe['source_manifest'],
             on_policy_prereg=recipe['on_policy_prereg'], original_group_indices=recipe['original_group_indices'],
             row_rule='all original unit rows, original order', gnn_literal=mod._GNN_LITERAL,
             first128_observation='Caller first h at return of own exact _fh90_first_head, before second SiLU overwrites h.',
             first_pending_cache='Python shape/data only, at most one live layer; no native tensor or frame retained.',
             limitation='Diagnostic fetches change request timing and materialization. Own plain/capture source anchors required; no runtime credit.')
        write(self.outdir/'METADATA_v1.json', self.metadata)
        sys.setprofile(self.profile)

    def abort(self, reason):
        # Restore first, including when capture validation or user code raises.
        was_open = not self.finished
        self.finished = True
        sys.setprofile(self.parent)
        self.pending.clear()
        if was_open:
            self.abort_reason = str(reason)

    def begin(self, rank, phase):
        try:
            assert not self.finished and not self.pending
            assert int(rank) in self.recipe['capture_ranks']
            assert str(phase) in ('warmup', 'scored')
            self.rank, self.phase = int(rank), str(phase)
        except BaseException:
            self.abort('begin failed')
            raise

    def fetch(self, value):
        return dict(shape=[int(x) for x in value.shape], data=value.tolist())

    def profile(self, frame, event, arg):
        try:
            self.observe(frame, event, arg)
        except BaseException:
            self.abort('profile failed')
            raise

    def observe(self, frame, event, arg):
        if self.parent is not None:
            self.parent(frame, event, arg)
        if self.finished or self.rank is None or event != 'return':
            return
        if frame.f_code in (self.first_code, self.code) and arg is None:
            self.abort('observed function failed or returned None')
            return
        if frame.f_code is self.first_code:
            caller = frame.f_back
            assert caller is not None and caller.f_code is self.code
            local, child = caller.f_locals, frame.f_locals
            layer = int(child['l'])
            assert 1 <= layer <= 15 and int(local['l']) == layer and child['k'] is local['k']
            key = (self.rank, self.phase, layer)
            assert key not in self.first_seen and key not in self.seen and not self.pending
            assert tuple(local['h'].shape) == (1024, 128) and str(local['h'].dtype) == 'float32'
            self.pending[key] = self.fetch(local['h'])
            self.first_seen.add(key)
            return
        if frame.f_code is not self.code:
            return
        local = frame.f_locals
        layer = int(local['l'])
        if layer == 0:
            assert not self.pending and 'h' not in local and 'k' not in local
            assert isinstance(arg, tuple) and len(arg) == 2
            self.skipped_layer0 += 1
            return
        assert 1 <= layer <= 15
        key = (self.rank, self.phase, layer)
        assert key not in self.seen and key in self.first_seen and set(self.pending) == {key}
        assert tuple(local['state'].shape) == (1024, 64) and tuple(local['h'].shape) == (1024, 128)
        assert tuple(local['k']['H'].shape) == (1024, 78)
        gate = local['k']['H'][:, 5]
        record = dict(rank=self.rank, phase=self.phase, layer=layer, first128=self.pending.pop(key),
                      state=self.fetch(local['state']), hidden2=self.fetch(local['h']),
                      sigma=self.fetch(local['sigma']), delta=self.fetch(local['delta']), gate_feature=self.fetch(gate))
        del gate
        path = self.outdir/('rank%04d_%s_L%02d.json.gz'%key)
        with gzip.open(path, 'xt', encoding='utf-8', compresslevel=3) as stream:
            json.dump(record, stream, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
        self.files.append(dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size,
                               rank=self.rank, phase=self.phase, layer=layer))
        self.seen.add(key)

    def finish(self):
        assert not self.finished
        self.finished = True
        sys.setprofile(self.parent)
        assert not self.pending and self.first_seen == self.seen
        assert self.gc_state == (gc.isenabled(), gc.get_threshold())
        assert self.alias_ids == {name: id(getattr(self.mod, name)) for name in self.alias_ids}
        groups = sorted({(rank, phase) for rank, phase, _ in self.seen})
        for rank, phase in groups:
            assert {layer for rr, pp, layer in self.seen if rr == rank and pp == phase} == set(range(1, 16))
        assert self.files and sum(row['bytes'] for row in self.files) < 1_000_000_000
        receipt = dict(utc=utc(), complete=True, groups=groups, files=self.files,
                       source_sha256=self.metadata['source_sha256'], target_free=True,
                       direct_first128_count=len(self.first_seen), all_first_paired_once=True,
                       native_handles_retained=False, native_handles_or_frames_retained=False, pending_cache_empty=True,
                       valid_layer0_returns_skipped=self.skipped_layer0,
                       source_function_aliases_unchanged=True, gc_settings_unchanged=True,
                       no_runtime_or_adjusted_score_claim=True)
        write(self.outdir/'CAPTURE_COMPLETE_v1.json', receipt)
        return receipt

def install(mod, outdir, recipe):
    return Observer(mod, outdir, recipe)
