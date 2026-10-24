"""Diagnostic-only, preallocated Python-function attribution outside SDK spans.

Construct the identical observer in both arms before prediction. The callback
creates no tuple/list/dict/set records: all stack and statistics cells are fixed
integer arrays, and code metadata is indexed before profiling begins. It never
changes SDK or GC policy and never retains frames, numerical arrays, arguments,
or bound methods. Direct C-call cost stays with its enclosing Python function.
"""
from array import array
import dis
import json
from pathlib import Path
import subprocess
import sys
import time
import types


def candidate_codes(filename, namespace):
    """Read callable/code metadata only; retain code objects, never callables."""
    pending = list(namespace.values())
    seen_objects = set()
    codes = {}
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen_objects:
            continue
        seen_objects.add(identity)
        if isinstance(value, types.FunctionType):
            code = value.__code__
            if code.co_filename == filename:
                codes[id(code)] = code
                pending.extend(code.co_consts)
                if value.__closure__:
                    for cell in value.__closure__:
                        try:
                            item = cell.cell_contents
                        except ValueError:
                            continue
                        if isinstance(item, types.FunctionType):
                            pending.append(item)
        elif isinstance(value, types.CodeType):
            if value.co_filename == filename:
                codes[identity] = value
                pending.extend(value.co_consts)
        elif isinstance(value, type):
            for item in vars(value).values():
                if isinstance(item, types.FunctionType):
                    pending.append(item)
                elif isinstance(item, (staticmethod, classmethod)):
                    pending.append(item.__func__)
                elif isinstance(item, property):
                    pending.extend(x for x in (item.fget, item.fset, item.fdel) if x is not None)
    return tuple(sorted(codes.values(), key=lambda code: (code.co_firstlineno, code.co_name, id(code))))


class OutsideDispatchProfile:
    def __init__(self, candidate, namespace, dispatch_reader=None, sdk_span_code=None):
        self.candidate = str(candidate)
        if dispatch_reader is None or sdk_span_code is None:
            from flopscope import _dispatch
            if dispatch_reader is None:
                dispatch_reader = _dispatch.total_dispatch_ns
            if sdk_span_code is None:
                sdk_span_code = _dispatch._counted_span.__wrapped__.__code__
        self.dispatch_reader = dispatch_reader
        self.sdk_span_code = sdk_span_code
        self.codes = candidate_codes(self.candidate, namespace)
        assert self.codes, 'candidate function metadata must be populated before prediction'
        self.code_index = {id(code): index for index, code in enumerate(self.codes)}
        # calls, wall, dispatch, outside, exclusive, negative exclusive calls
        self.stats = array('q', [0]) * (6 * len(self.codes))
        # code index, entry wall, entry dispatch, child outside
        self.stack = array('q', [0]) * (4 * 2048)
        self.depth = 0
        self.sdk_stack = array('Q', [0]) * 128
        self.sdk_depth = 0
        self.root_wall_ns = 0
        self.root_dispatch_ns = 0
        self.root_outside_ns = 0
        self.calls = 0
        self.returns = 0
        self.excluded_candidate_events_in_sdk = 0
        self.running = False
        self.previous = None
        self.started = False
        self.max_depth = 0
        self.max_sdk_depth = 0
        self.yield_opcode = dis.opmap['YIELD_VALUE']
        self.resume_opcode = dis.opmap.get('RESUME', -1)

    def __call__(self, frame, event, arg):
        # Metadata-only callback. No per-event tracked container allocations.
        if event != 'call' and event != 'return':
            return
        code = frame.f_code
        if code is self.sdk_span_code and event == 'return':
            opcode = code.co_code[frame.f_lasti]
            yielded = opcode == self.yield_opcode
            if opcode == self.resume_opcode and frame.f_lasti >= 2:
                yielded = code.co_code[frame.f_lasti - 2] == self.yield_opcode
            if yielded:
                if self.sdk_depth >= 128:
                    raise RuntimeError('preallocated SDK span stack exceeded')
                self.sdk_stack[self.sdk_depth] = id(frame)
                self.sdk_depth += 1
                if self.sdk_depth > self.max_sdk_depth:
                    self.max_sdk_depth = self.sdk_depth
            else:
                if self.sdk_depth == 0 or self.sdk_stack[self.sdk_depth - 1] != id(frame):
                    raise RuntimeError('SDK span nesting mismatch')
                self.sdk_depth -= 1
            return
        if code.co_filename != self.candidate:
            return
        if self.sdk_depth:
            self.excluded_candidate_events_in_sdk += 1
            return
        index = self.code_index.get(id(code), -1)
        if index < 0:
            raise RuntimeError('unregistered candidate code object')
        now = time.perf_counter_ns()
        dispatched = self.dispatch_reader()
        if event == 'call':
            self.calls += 1
            if self.depth >= 2048:
                raise RuntimeError('preallocated candidate stack exceeded')
            at = self.depth * 4
            self.stack[at] = index
            self.stack[at + 1] = now
            self.stack[at + 2] = dispatched
            self.stack[at + 3] = 0
            self.depth += 1
            if self.depth > self.max_depth:
                self.max_depth = self.depth
            return
        self.returns += 1
        if self.depth == 0:
            raise RuntimeError('unmatched candidate return')
        self.depth -= 1
        at = self.depth * 4
        if self.stack[at] != index:
            raise RuntimeError('candidate stack mismatch')
        wall = now - self.stack[at + 1]
        dispatch = dispatched - self.stack[at + 2]
        outside = wall - dispatch
        exclusive = outside - self.stack[at + 3]
        stat = index * 6
        self.stats[stat] += 1
        self.stats[stat + 1] += wall
        self.stats[stat + 2] += dispatch
        self.stats[stat + 3] += outside
        self.stats[stat + 4] += exclusive
        self.stats[stat + 5] += exclusive < 0
        if self.depth:
            self.stack[at - 1] += outside
        else:
            self.root_wall_ns += wall
            self.root_dispatch_ns += dispatch
            self.root_outside_ns += outside

    def start(self):
        assert not self.running and not self.started and self.depth == self.sdk_depth == 0
        self.previous = sys.getprofile()
        assert self.previous is None, 'do not displace another profiler'
        self.running = True
        self.started = True
        sys.setprofile(self)

    def stop(self):
        if self.running:
            sys.setprofile(self.previous)
            self.running = False
        assert self.depth == 0 and self.sdk_depth == 0, 'all candidate and SDK frames must retire'

    def report(self):
        self.stop()
        rows = []
        for index, code in enumerate(self.codes):
            at = index * 6
            if self.stats[at]:
                rows.append(dict(kind='python_with_direct_c', line=code.co_firstlineno,
                    function=code.co_name, calls=self.stats[at], wall_inclusive_ns=self.stats[at+1],
                    dispatch_inclusive_ns=self.stats[at+2], outside_inclusive_ns=self.stats[at+3],
                    outside_exclusive_ns=self.stats[at+4], negative_exclusive_calls=self.stats[at+5]))
        rows.sort(key=lambda row: row['outside_exclusive_ns'], reverse=True)
        assert sum(row['outside_exclusive_ns'] for row in rows) == self.root_outside_ns
        return dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
            diagnostic_only=True, candidate=self.candidate, started=self.started,
            events=dict(call=self.calls, return_event=self.returns),
            registered_code_objects=len(self.codes), max_depth=self.max_depth, max_sdk_depth=self.max_sdk_depth,
            excluded_candidate_events_in_sdk=self.excluded_candidate_events_in_sdk,
            root_wall_ns=self.root_wall_ns, root_dispatch_ns=self.root_dispatch_ns,
            root_outside_ns=self.root_outside_ns, rows=rows, stack_empty=True, sdk_unmodified=True,
            gc_policy_unmodified=True, preallocated_integer_storage=True,
            no_frames_arguments_numerical_arrays_or_bound_methods_retained=True,
            limitation='Instrumentation still perturbs timing. No causal/runtime qualification. Direct C-call cost is included in the enclosing Python function. Strict own-control prediction, bill, operations, requests and free-count parity required.')

    def write(self, path):
        report = self.report()
        with Path(path).open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        return report
