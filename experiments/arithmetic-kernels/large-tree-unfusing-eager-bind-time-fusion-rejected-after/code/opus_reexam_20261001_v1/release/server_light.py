"""TAG opus-release: the stock flopscope server plus passive counters.

1. Peak handle count (one integer compare per stored array), the number of arrays stored and their logical bytes by number
   type (shape x item size, views counted in full: a fingerprint of every result's shape and type for fast_path.py).
2. Kernel work by operation: every kernel call the server times (RequestHandler._run_kernel, the span the platform reports as
   flopscope_server_kernel_wall_s) is bracketed by two kernel-counter reads (rusage.raw, about 3.4 microseconds per request,
   booked OUTSIDE the timed span), giving wall nanoseconds, cycles and instructions per operation name.
3. The numpy part of that work: the span also contains flopscope's own per-request wrapper (billing, records, locks: about 0.8
   million instructions per request). The numpy calls inside it (flopscope's single choke point _call_numpy_impl) are bracketed
   as well (another 3.4 microseconds per request, inside the span), so wrapper = kernel - numpy is known per network.
4. Wire work: the server's own "comms" time per request (decoding the request + encoding the reply; the platform reports its sum
   per network and per setup as flopscope_server_comms_wall_s / comms_ns) summed with request and byte counts, and the same
   two steps bracketed by counter reads (about 10 microseconds per request, outside the kernel span), so the wire work per
   request is known in instructions as well as in nanoseconds.
Cycles and instructions do not move with CPU load the way wall time does, but they read high when the process pages (rusage.py).
On SIGUSR1 the server appends one JSON line of cumulative totals to $RELEASE_SERVER_MARKS (the client marks network boundaries
with it). At exit or SIGTERM it writes $RELEASE_SERVER_RECEIPT. Nothing else in the serving path changes.
usage (OpenBLAS env python): server_light.py --url ipc:///private/tmp/x.sock --timeout 1800"""
import atexit
import json
import os
import resource
import signal
import sys
from time import perf_counter_ns

from flopscope import _budget
from flopscope_server import _array_store, _comms_tracker, _request_handler, _server

import rusage

receipt = os.environ['RELEASE_SERVER_RECEIPT']
marks = os.environ.get('RELEASE_SERVER_MARKS')
state = dict(peak=0, puts=0)
put_bytes = {}                 # number type ('<f4', '<f8', ...) -> logical bytes of every array stored
ops = {}                       # operation name -> [calls, kernel ns, cycles, instructions]
numpy_ops = {}                 # numpy function name -> [calls, ns, cycles, instructions] (inside the kernel spans)
comms = [0, 0, 0, 0]           # requests, comms ns (the server's own measurement), bytes received, bytes sent
comms_work = [0, 0, 0]         # bracketed decode / normalise / encode calls, cycles, instructions
_impl = _budget._call_numpy_impl
_record = _comms_tracker.CommsTracker.record_request
_put = _array_store.ArrayStore.put
_raw = rusage.raw


def put(self, arr):
    handle = _put(self, arr)
    n = len(self._arrays)
    if n > state['peak']:
        state['peak'] = n
    state['puts'] += 1
    kind = getattr(getattr(arr, 'dtype', None), 'str', 'other')
    put_bytes[kind] = put_bytes.get(kind, 0) + getattr(arr, 'nbytes', 0)
    return handle


def run_kernel(self, fn, *args, **kwargs):
    c0, i0 = _raw()
    t0 = perf_counter_ns()
    try:
        return fn(*args, **kwargs)
    finally:
        dt = perf_counter_ns() - t0
        self._kernel_ns += dt
        c1, i1 = _raw()
        name = getattr(fn, '__name__', None) or type(fn).__name__
        row = ops.get(name)
        if row is None:
            row = ops[name] = [0, 0, 0, 0]
        row[0] += 1
        row[1] += dt
        row[2] += c1 - c0
        row[3] += i1 - i0


def call_numpy(fn, args, kwargs, *, track_python_callbacks):
    c0, i0 = _raw()
    t0 = perf_counter_ns()
    try:
        return _impl(fn, args, kwargs, track_python_callbacks=track_python_callbacks)
    finally:
        dt = perf_counter_ns() - t0
        c1, i1 = _raw()
        name = getattr(fn, '__name__', None) or type(fn).__name__
        row = numpy_ops.get(name)
        if row is None:
            row = numpy_ops[name] = [0, 0, 0, 0]
        row[0] += 1
        row[1] += dt
        row[2] += c1 - c0
        row[3] += i1 - i0


def record_request(self, *, bytes_received, bytes_sent, comms_overhead_ns, compute_time_ns, is_fetch):
    comms[0] += 1
    comms[1] += comms_overhead_ns
    comms[2] += bytes_received
    comms[3] += bytes_sent
    return _record(self, bytes_received=bytes_received, bytes_sent=bytes_sent, comms_overhead_ns=comms_overhead_ns,
                   compute_time_ns=compute_time_ns, is_fetch=is_fetch)


def bracket(fn):
    def wrapped(*args, **kwargs):
        c0, i0 = _raw()
        try:
            return fn(*args, **kwargs)
        finally:
            c1, i1 = _raw()
            comms_work[0] += 1
            comms_work[1] += c1 - c0
            comms_work[2] += i1 - i0
    return wrapped


def snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(peak_handles=state['peak'], puts=state['puts'], put_bytes=dict(put_bytes),
                array_count_limit=_array_store.MAX_ARRAY_COUNT,
                max_rss_bytes=usage.ru_maxrss, minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt,
                kernel_calls=sum(v[0] for v in ops.values()), kernel_s=sum(v[1] for v in ops.values())/1e9,
                kernel_cycles=sum(v[2] for v in ops.values()), kernel_instructions=sum(v[3] for v in ops.values()),
                ops={k: list(v) for k, v in ops.items()}, numpy_ops={k: list(v) for k, v in numpy_ops.items()},
                comms=list(comms), comms_work=list(comms_work),
                **rusage.read())


def mark(*_):
    with open(marks, 'a') as stream:
        stream.write(json.dumps(snapshot()) + '\n')


def write():
    with open(receipt, 'w') as stream:
        json.dump(snapshot(), stream)


_array_store.ArrayStore.put = put
_request_handler.RequestHandler._run_kernel = run_kernel
_budget._call_numpy_impl = call_numpy
_comms_tracker.CommsTracker.record_request = record_request
for _name in ('decode_request', '_normalize_msg', 'encode_response'):   # the two spans the server books as comms
    setattr(_server, _name, bracket(getattr(_server, _name)))
atexit.register(write)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
if marks:
    signal.signal(signal.SIGUSR1, mark)

from flopscope_server.__main__ import main  # noqa: E402

main()
