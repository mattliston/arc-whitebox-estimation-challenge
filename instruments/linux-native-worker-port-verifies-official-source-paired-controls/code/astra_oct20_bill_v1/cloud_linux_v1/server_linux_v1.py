"""Stock0.12.3 server with passive handles/comms marks; no kernel/tariff patch."""
import atexit
import json
import os
from pathlib import Path
import signal
import sys
from common_linux_v1 import check_sdk, load_plan, own_usage

root = Path(os.environ['ARC_BUNDLE_ROOT'])
plan = load_plan(root, os.environ['ARC_RUN_PLAN'])
environment = check_sdk(root, plan, 'server')
from flopscope_server import _array_store, _comms_tracker

state = dict(peak_handles=0, puts=0, put_bytes={})
comms = [0, 0, 0, 0]
_put = _array_store.ArrayStore.put
_record = _comms_tracker.CommsTracker.record_request


def put(self, arr):
    handle = _put(self, arr)
    state['peak_handles'] = max(state['peak_handles'], len(self._arrays))
    state['puts'] += 1
    kind = getattr(getattr(arr, 'dtype', None), 'str', 'other')
    state['put_bytes'][kind] = state['put_bytes'].get(kind, 0) + getattr(arr, 'nbytes', 0)
    return handle


def record_request(self, *, bytes_received, bytes_sent, comms_overhead_ns, compute_time_ns, is_fetch):
    comms[0] += 1; comms[1] += comms_overhead_ns; comms[2] += bytes_received; comms[3] += bytes_sent
    return _record(self, bytes_received=bytes_received, bytes_sent=bytes_sent,
                   comms_overhead_ns=comms_overhead_ns, compute_time_ns=compute_time_ns, is_fetch=is_fetch)


def snapshot():
    return dict(**state, comms=list(comms), process=own_usage(), environment=environment,
                array_count_limit=_array_store.MAX_ARRAY_COUNT, hardware_counters_available=False)


def mark(*_):
    with open(os.environ['RELEASE_SERVER_MARKS'], 'a') as f:
        f.write(json.dumps(snapshot()) + '\n')


def finish():
    Path(os.environ['RELEASE_SERVER_RECEIPT']).write_text(json.dumps(snapshot(), indent=2) + '\n')


_array_store.ArrayStore.put = put
_comms_tracker.CommsTracker.record_request = record_request
atexit.register(finish)
signal.signal(signal.SIGUSR1, mark)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
from flopscope_server.__main__ import main
main()
