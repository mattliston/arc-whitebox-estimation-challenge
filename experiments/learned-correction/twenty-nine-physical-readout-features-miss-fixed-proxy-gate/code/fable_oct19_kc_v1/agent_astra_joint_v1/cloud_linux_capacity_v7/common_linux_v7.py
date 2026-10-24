"""Thread-explicit Linux science helpers. Official numerics/tariff unchanged."""
import os
from pathlib import Path
import sys
import hashlib

PORT_V1 = Path(__file__).resolve().parents[3] / 'astra_oct20_bill_v1/cloud_linux_v1'
assert hashlib.sha256((PORT_V1 / 'common_linux_v1.py').read_bytes()).hexdigest() == '13aead70dfce132335df80829c63bbfacc039d823dd67b9276746e7046004f37'
assert hashlib.sha256((PORT_V1 / 'guard_entry_linux_v1.py').read_bytes()).hexdigest() == 'bdf733c338dd5d5ecf83643cea4a0a202f13f2dd446f12b059038d9a50cf00d1'
sys.path.insert(0, str(PORT_V1))
from common_linux_v1 import *
from common_linux_v1 import check_sdk as _check_sdk_v1

THREAD_KEYS = tuple(THREADS)

def thread_env(n):
    n = int(n)
    assert n in (1, 2, 4), 'only frozen BLAS1/2/4 research conditions'
    return {key: str(n) for key in THREAD_KEYS}

THREADS = thread_env(os.environ.get('ARC_BLAS_THREADS', '1'))

def thread_provenance():
    n = int(os.environ['ARC_BLAS_THREADS'])
    actual = {key: os.environ.get(key) for key in THREAD_KEYS}
    assert actual == thread_env(n), ('thread environment drift', actual, n)
    return dict(blas_threads=n, environment=actual, logical_cpu_count=os.cpu_count(),
                configured_not_assumed_utilization=True)

def check_sdk(root, plan, role):
    threads = thread_provenance()
    result = _check_sdk_v1(root, plan, role)
    result['thread_provenance'] = threads
    return result
