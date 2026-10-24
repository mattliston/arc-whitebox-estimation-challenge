"""Diagnostic only: source-site time after subtracting official SDK dispatch.

Observes Python and direct-C call boundaries from one exact candidate filename.
It changes neither SDK implementation nor dispatch accounting, and stores no
frames, argument values, bound methods or arrays. Its own overhead means these
times cannot qualify runtime. Fresh unobserved control/bytes/bill/call checks
must accompany any whole-model use.
"""
import collections
import dis
import json
from pathlib import Path
import subprocess
import sys
import time


class OutsideDispatchProfile:
    def __init__(self, candidate, dispatch_reader=None):
        self.candidate = str(candidate)
        if dispatch_reader is None:
            from flopscope._dispatch import total_dispatch_ns
            dispatch_reader = total_dispatch_ns
        self.dispatch_reader = dispatch_reader
        self.stack = []
        self.stats = {}
        self.events = collections.Counter()
        self.root_outside_ns = 0
        self.root_wall_ns = 0
        self.root_dispatch_ns = 0
        self.running = False
        self.previous = None
        self.active_sdk_spans = set()
        self.excluded_candidate_events_in_sdk = 0

    def _trace(self, frame, event, arg):
        if (frame.f_globals.get('__name__') == 'flopscope._dispatch'
                and frame.f_code.co_name == '_counted_span' and event == 'return'):
            # The official generator yields while a dispatch span is active.
            # Track identities only, so candidate destructors entered by SDK
            # work remain attributed to dispatch instead of counted twice.
            op = frame.f_code.co_code[frame.f_lasti]
            yielded = op == dis.opmap['YIELD_VALUE']
            # Python3.13 reports the following RESUME position on generator
            # yield; the pinned VM3.10 reports YIELD_VALUE itself.
            if op == dis.opmap.get('RESUME') and frame.f_lasti >= 2:
                yielded = frame.f_code.co_code[frame.f_lasti - 2] == dis.opmap['YIELD_VALUE']
            if yielded:
                self.active_sdk_spans.add(id(frame))
            else:
                self.active_sdk_spans.discard(id(frame))
            return
        if frame.f_code.co_filename != self.candidate:
            return
        if self.active_sdk_spans:
            self.excluded_candidate_events_in_sdk += 1
            return
        if event not in ('call', 'return', 'c_call', 'c_return', 'c_exception'):
            return
        now = time.perf_counter_ns()
        dispatched = self.dispatch_reader()
        self.events[event] += 1
        if event in ('call', 'return'):
            key = ('python', frame.f_code.co_firstlineno, frame.f_code.co_name)
        else:
            # Only ordinary builtin function metadata is read. Never retain arg.
            key = ('direct_c', frame.f_lineno, getattr(arg, '__qualname__', type(arg).__name__))
        if event in ('call', 'c_call'):
            self.stack.append([key, now, dispatched, 0])
            return
        assert self.stack, ('unmatched candidate return', key, event)
        entry = self.stack.pop()
        assert entry[0] == key, ('candidate stack mismatch', entry[0], key, event)
        wall = now - entry[1]
        dispatch = dispatched - entry[2]
        outside = wall - dispatch
        exclusive = outside - entry[3]
        row = self.stats.setdefault(key, dict(calls=0, wall_inclusive_ns=0,
                    dispatch_inclusive_ns=0, outside_inclusive_ns=0,
                    outside_exclusive_ns=0, negative_exclusive_calls=0))
        row['calls'] += 1
        row['wall_inclusive_ns'] += wall
        row['dispatch_inclusive_ns'] += dispatch
        row['outside_inclusive_ns'] += outside
        row['outside_exclusive_ns'] += exclusive
        row['negative_exclusive_calls'] += exclusive < 0
        if self.stack:
            self.stack[-1][3] += outside
        else:
            self.root_wall_ns += wall
            self.root_dispatch_ns += dispatch
            self.root_outside_ns += outside

    def start(self):
        assert not self.running and not self.stack and not self.active_sdk_spans
        self.previous = sys.getprofile()
        assert self.previous is None, 'do not displace another profiler'
        self.running = True
        sys.setprofile(self._trace)

    def stop(self):
        if self.running:
            sys.setprofile(self.previous)
            self.running = False
        assert not self.stack, 'candidate call stack must retire before profile stop'
        assert not self.active_sdk_spans, 'official dispatch spans must retire before profile stop'

    def write(self, path):
        self.stop()
        rows = [dict(kind=key[0], line=key[1], function=key[2], **value)
                for key, value in self.stats.items()]
        rows.sort(key=lambda row: row['outside_exclusive_ns'], reverse=True)
        assert sum(row['outside_exclusive_ns'] for row in rows) == self.root_outside_ns
        report = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
                      diagnostic_only=True, candidate=self.candidate,
                      events=dict(self.events), root_wall_ns=self.root_wall_ns,
                      excluded_candidate_events_in_sdk=self.excluded_candidate_events_in_sdk,
                      root_dispatch_ns=self.root_dispatch_ns,
                      root_outside_ns=self.root_outside_ns,
                      rows=rows, stack_empty=True, sdk_unmodified=True,
                      no_frames_arguments_arrays_or_bound_methods_retained=True,
                      limitation='Instrumentation overhead perturbs all observed calls. SDK counter is read only; subtraction attributes current official dispatch, not a causal hosted-time forecast. Negative per-call residuals are retained, never clamped. Only predict is observed; no setup/fetch/held truth.')
        with Path(path).open('x') as stream:
            json.dump(report, stream, indent=2); stream.write('\n')
        return report
