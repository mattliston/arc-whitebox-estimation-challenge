"""Diagnostic C-level Python self-time attribution; no SDK/GC modification.

Both arms construct this identical state before prediction; only the observed
arm enables CPython's standard C profiler. Python self times exclude callees,
including all SDK functions. They are a lower-level hotspot inventory, not
per-function wall-minus-dispatch measurements or a release timing gate.
"""
import cProfile
import json
from pathlib import Path
import subprocess
import sys
import time
import types


class CandidateProfile:
    def __init__(self, candidate, dispatch_reader=None):
        self.candidate = str(candidate)
        if dispatch_reader is None:
            from flopscope._dispatch import total_dispatch_ns
            dispatch_reader = total_dispatch_ns
        self.dispatch_reader = dispatch_reader
        self.profiler = cProfile.Profile()
        assert sys.getprofile() is None, 'do not displace another profiler'
        # Identical metadata-only initialization in both arms, before predict.
        self.profiler.enable()
        self.profiler.disable()
        self.profiler.clear()
        self.started = False
        self.running = False
        self.wall_ns = 0
        self.dispatch_ns = 0
        self.start_wall = 0
        self.start_dispatch = 0

    def start(self):
        assert not self.started and not self.running
        assert sys.getprofile() is None, 'do not displace another profiler'
        self.start_dispatch = self.dispatch_reader()
        self.start_wall = time.perf_counter_ns()
        self.started = self.running = True
        self.profiler.enable()

    def stop(self):
        if self.running:
            self.profiler.disable()
            self.wall_ns = time.perf_counter_ns() - self.start_wall
            self.dispatch_ns = self.dispatch_reader() - self.start_dispatch
            self.running = False
        assert sys.getprofile() is None, 'profiler must retire before output'

    def report(self):
        self.stop()
        rows = []
        sdk_rows = []
        for entry in self.profiler.getstats():
            code = entry.code
            if not isinstance(code, types.CodeType):
                continue
            if code.co_filename == self.candidate:
                direct_c = []
                for subentry in entry.calls or ():
                    if isinstance(subentry.code, str):
                        direct_c.append(dict(function=subentry.code, calls=subentry.callcount,
                            self_s=subentry.inlinetime, cumulative_s=subentry.totaltime))
                direct_c.sort(key=lambda row: row['self_s'], reverse=True)
                rows.append(dict(line=code.co_firstlineno, function=code.co_name,
                    calls=entry.callcount, recursive_calls=entry.reccallcount,
                    self_s=entry.inlinetime, cumulative_s=entry.totaltime,
                    direct_c_children=direct_c,
                    direct_c_self_s=sum(row['self_s'] for row in direct_c)))
            elif '/flopscope/' in code.co_filename:
                sdk_rows.append(dict(file=code.co_filename, line=code.co_firstlineno,
                    function=code.co_name, calls=entry.callcount,
                    self_s=entry.inlinetime, cumulative_s=entry.totaltime))
        rows.sort(key=lambda row: row['self_s'], reverse=True)
        sdk_rows.sort(key=lambda row: row['self_s'], reverse=True)
        return dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
            diagnostic_only=True, candidate=self.candidate, started=self.started,
            method='CPython standard C-level cProfile; default C timer; no Python event callback',
            root_wall_ns=self.wall_ns, root_dispatch_ns=self.dispatch_ns,
            root_outside_ns=self.wall_ns-self.dispatch_ns,
            candidate_python_self_s=sum(row['self_s'] for row in rows),
            candidate_direct_c_self_s=sum(row['direct_c_self_s'] for row in rows),
            rows=rows, sdk_self_time_inventory=sdk_rows, stack_empty=True,
            sdk_unmodified=True, gc_policy_unmodified=True, profiler_restored=True,
            limitations=['Timing is perturbed and cannot qualify runtime.',
                'Candidate self times exclude all Python/C callees; direct C edges are separately shown.',
                'SDK self-time inventory mixes counted and uncounted work; never subtract or aggregate it as residual.',
                'Only root outside time subtracts the read-only official dispatch counter.',
                'Require exact own-control prediction, FLOPs, operations, requests, free trips and freed handles.'])

    def write(self, path):
        report = self.report()
        with Path(path).open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        return report
