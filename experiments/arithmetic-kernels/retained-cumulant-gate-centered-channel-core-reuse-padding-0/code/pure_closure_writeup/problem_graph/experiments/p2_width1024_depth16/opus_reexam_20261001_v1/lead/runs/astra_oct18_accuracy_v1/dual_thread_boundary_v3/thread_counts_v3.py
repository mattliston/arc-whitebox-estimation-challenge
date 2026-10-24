"""Diagnostic-only Darwin current-thread counters; never imported by submission code.
ABI: sys/resource_private.h THSC_CPI=1, struct thsc_cpi {u64 instructions,u64 cycles};
sys/syscall.h SYS_thread_selfcounts=186; sys/sysproto.h (uint32_t, user_addr_t, user_size_t).
unistd.h declares int syscall(int, ...); one fixed arg is essential on Darwin ARM64.
"""
import ctypes
import os
import sys
import threading

assert sys.platform == 'darwin', 'thread_selfcounts diagnostic requires Darwin'
class _CPI(ctypes.Structure):
    _fields_ = [('instructions', ctypes.c_uint64), ('cycles', ctypes.c_uint64)]
assert ctypes.sizeof(_CPI) == 16
_libc = ctypes.CDLL(None, use_errno=True)
_syscall = _libc.syscall
_syscall.argtypes = (ctypes.c_int,)
_syscall.restype = ctypes.c_int
_buf = _CPI()
_ptr = ctypes.byref(_buf)
_kind = ctypes.c_uint32(1)
_size = ctypes.c_size_t(ctypes.sizeof(_buf))
_owner = threading.get_ident()
_native_owner = threading.get_native_id()

def raw():
    # Check before touching shared storage; reject any unexpected off-thread dispatch.
    assert threading.get_ident() == _owner, "counter collector changed threads"
    # Explicitly typed variadic arguments.
    rc = _syscall(186, _kind, _ptr, _size)
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, 'thread_selfcounts: ' + os.strerror(err))
    return _buf.cycles, _buf.instructions

def assert_owner():
    assert threading.get_ident() == _owner, 'counter collector changed threads'
    assert threading.get_native_id() == _native_owner, 'counter collector changed native threads'

def selfcheck():
    assert_owner()
    a, b = raw(), raw()
    assert all(v > 0 for v in a), ('unavailable thread counters', a)
    assert all(y >= x for x,y in zip(a,b)), ('nonmonotonic thread counters', a,b)
    return dict(syscall=186, kind=1, structure_bytes=16, structure_fields=['instructions','cycles'],
                returned_order=['cycles','instructions'], thread_ident=_owner, native_thread_id=_native_owner,
                startup_counts=list(b), diagnostic_only=True, payload_unchanged=True,
                process_counter_scope='all process threads', thread_counter_scope='measurement thread only',
                comparison_requires_contemporaneous_same_harness_controls=True,
                dispatch_window='thread outer, process inner', top_level_span_count_recorded=True)
