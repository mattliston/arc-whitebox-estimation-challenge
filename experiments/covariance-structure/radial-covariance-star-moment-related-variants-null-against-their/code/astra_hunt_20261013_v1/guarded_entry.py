"""Single-process numerical entry; invoked as guard_footprint_v3's child.

Checks nice and CPU/thread constraints, gates work on memory_pressure >= 50%,
and locks this hunt's numerical lane. All records live next to this file.
"""
import fcntl
import ctypes
import gc
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
lock = (HERE / 'numerical_lane.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice 10 required'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'VECLIB_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(name) == '1', (name, os.environ.get(name))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

record = HERE / (os.environ['ASTRA_PRESSURE_RECORD'])
assert not record.exists()
last_pressure = 0.0
last_relief = 0.0
libsystem = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
allocator_relief = libsystem.malloc_zone_pressure_relief
allocator_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
allocator_relief.restype = ctypes.c_size_t


def relieve_unused_pages():
    global last_relief
    if os.environ.get('ASTRA_MEMORY_RELIEF', '0') != '1':
        return
    if time.monotonic() - last_relief >= 5:
        gc.collect()
        allocator_relief(None, 0)
        last_relief = time.monotonic()


def check_pressure():
    global last_pressure
    if time.monotonic() - last_pressure < 10:
        return
    while True:
        text = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
        pct = int(re.search(r'System-wide memory free percentage: (\d+)%', text).group(1))
        with record.open('a') as stream:
            stream.write(json.dumps({'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                     'free_pct': pct, 'pid': os.getpid()}) + '\n')
        if pct >= 50:
            last_pressure = time.monotonic()
            return
        print('ASTRA pressure gate waiting:', pct, flush=True)
        time.sleep(10)


check_pressure()
target, *args = sys.argv[1:]
if Path(target).name == 'run_cert_v2.py':
    from flopscope import _budget
    original_charge = _budget.BudgetContext._charge_op
    observed_layer = [None]

    def pressure_charge(self, *a, **kw):
        check_pressure()
        relieve_unused_pages()
        candidate = sys.modules.get('candidate_cert')
        if candidate is not None:
            layer = getattr(candidate, '_PHYS_LAYER', [None])[0]
            if layer != observed_layer[0]:
                observed_layer[0] = layer
                print(json.dumps({'astra_layer': layer, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}), flush=True)
        return original_charge(self, *a, **kw)

    _budget.BudgetContext._charge_op = pressure_charge

sys.argv = [target, *args]
sys.path.insert(0, str(Path(target).resolve().parent))
runpy.run_path(target, run_name='__main__')
if Path(target).name == 'run_cert_v2.py' and os.environ.get('ASTRA_BASE_CERT'):
    comparison = str(Path(args[-1]).with_suffix('')) + '_comparison.json'
    sys.argv = [str(HERE / 'compare_cert.py'), os.environ['ASTRA_BASE_CERT'], args[-1], comparison]
    runpy.run_path(str(HERE / 'compare_cert.py'), run_name='__main__')
    if os.environ.get('ASTRA_SECOND_BASE_CERT'):
        secondary = Path(os.environ['ASTRA_SECOND_BASE_CERT']).resolve()
        secondary.relative_to(HERE)
        comparison = str(Path(args[-1]).with_suffix('')) + '_secondary_comparison.json'
        sys.argv = [str(HERE / 'compare_cert.py'), str(secondary), args[-1], comparison]
        runpy.run_path(str(HERE / 'compare_cert.py'), run_name='__main__')
