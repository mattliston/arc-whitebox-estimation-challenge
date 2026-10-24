"""Observe real ArrayStore high-water counts without retaining any arrays.

The estimator and installed server files are unchanged. This instrumentation
is for handle/resource qualification, not runtime comparison.
"""
import atexit
import json
import os
from pathlib import Path
import signal
import time
from flopscope_server import _array_store

target = Path(os.environ['CODEX_HANDLE_RECEIPT'])
assert not target.exists()
records = []
last_write = [0.0]
original_init = _array_store.ArrayStore.__init__
original_put = _array_store.ArrayStore.put
original_free = _array_store.ArrayStore.free
original_clear = _array_store.ArrayStore.clear

def write(force=False):
    now = time.monotonic()
    if force or now - last_write[0] >= 1.0:
        data = dict(array_count_limit=_array_store.MAX_ARRAY_COUNT, stores=records,
                    peak_handles=max((r['peak'] for r in records), default=0),
                    observed_monotonic=now)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(target)
        last_write[0] = now

def observed_init(self):
    original_init(self)
    self._codex_census_id = len(records)
    records.append(dict(current=0, peak=0, puts=0, frees=0, clears=0))

def observed_put(self, arr):
    handle = original_put(self, arr)
    record = records[self._codex_census_id]
    record['current'] = self.count
    record['peak'] = max(record['peak'], self.count)
    record['puts'] += 1
    write()
    return handle

def observed_free(self, handles):
    original_free(self, handles)
    record = records[self._codex_census_id]
    record['current'] = self.count
    record['frees'] += len(handles)
    write()

def observed_clear(self):
    original_clear(self)
    record = records[self._codex_census_id]
    record['current'] = self.count
    record['clears'] += 1
    write()

_array_store.ArrayStore.__init__ = observed_init
_array_store.ArrayStore.put = observed_put
_array_store.ArrayStore.free = observed_free
_array_store.ArrayStore.clear = observed_clear
atexit.register(lambda: write(True))
def stop(signum, frame):
    write(True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)

from flopscope_server.__main__ import main
main()
