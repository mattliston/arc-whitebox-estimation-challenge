"""Diagnostic-only client coercion observer; never use its timings for release gates.

Delegates every operation unchanged. Does not enter, reset, or mutate a dispatch
span, fetch any extra value, or retain a backend array. Inclusive nested timings
are separate from top-level timings, so bool -> fetch is not summed twice.
"""
import functools
import json
from pathlib import Path
import sys
import time


class SyncObserver:
    METHODS = ('_fetch_data', '__bool__', '__float__', '__int__', '__index__',
               '__complex__', '__array__', '__contains__', 'item', 'tolist',
               '__repr__', '__str__', '__len__', '__iter__')

    def __init__(self, candidate_filename, dispatch_total=None):
        self.candidate_filename = str(candidate_filename)
        self.phase_name, self.rank = 'harness', None
        self.rows, self.originals, self.stack = {}, [], []
        if dispatch_total is None:
            from flopscope import _dispatch
            dispatch_total = _dispatch.total_dispatch_ns
        self.dispatch_total = dispatch_total

    def phase(self, name, rank=None):
        if self.stack:
            raise RuntimeError('observer phase changed during a conversion')
        self.phase_name, self.rank = str(name), rank

    def _site(self):
        frame = sys._getframe(2)
        while frame is not None:
            if frame.f_code.co_filename == self.candidate_filename:
                return f'{frame.f_code.co_name}:{frame.f_lineno}'
            frame = frame.f_back
        return '<harness>'

    def _instrument(self, cls, method):
        original = getattr(cls, method, None)
        if original is None:
            return
        owner = self
        label = cls.__name__ + '.' + method

        @functools.wraps(original)
        def observed(obj, *args, **kwargs):
            site = owner._site()
            top = not owner.stack
            key = (owner.phase_name, owner.rank, label, site, top)
            record = owner.rows.get(key)
            if record is None:
                record = owner.rows[key] = dict(calls=0, exceptions=0, wall_ns=0,
                                                dispatch_ns=0, outside_dispatch_ns=0)
            record['calls'] += 1
            owner.stack.append(label)
            d0, t0 = owner.dispatch_total(), time.perf_counter_ns()
            failed = False
            try:
                return original(obj, *args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                elapsed = time.perf_counter_ns() - t0
                dispatch = owner.dispatch_total() - d0
                owner.stack.pop()
                record['exceptions'] += int(failed)
                record['wall_ns'] += elapsed
                record['dispatch_ns'] += dispatch
                record['outside_dispatch_ns'] += elapsed - dispatch
        self.originals.append((cls, method, original))
        setattr(cls, method, observed)

    def install(self, classes=None):
        if self.originals:
            raise RuntimeError('observer already installed')
        if classes is None:
            from flopscope._remote_array import RemoteArray, RemoteScalar
            classes = (RemoteArray, RemoteScalar)
        for cls in classes:
            for method in self.METHODS:
                self._instrument(cls, method)

    def uninstall(self):
        for cls, method, original in reversed(self.originals):
            setattr(cls, method, original)
        self.originals.clear()

    def snapshot(self):
        records = [dict(phase=key[0], rank=key[1], method=key[2], site=key[3],
                        top_level=key[4], **values)
                   for key, values in self.rows.items()]
        records.sort(key=lambda r: (r['phase'], -1 if r['rank'] is None else r['rank'],
                                    r['site'], r['method'], r['top_level']))
        return dict(diagnostic_only=True, candidate=self.candidate_filename,
                    timing_note='Observer adds Python overhead. Nested inclusive timings overlap. '
                                'Use only top_level rows for a nonduplicated elapsed sum. '
                                'These values must not enter qualification projections.',
                    records=records)

    def write(self, path):
        Path(path).write_text(json.dumps(self.snapshot(), indent=2) + '\n')
