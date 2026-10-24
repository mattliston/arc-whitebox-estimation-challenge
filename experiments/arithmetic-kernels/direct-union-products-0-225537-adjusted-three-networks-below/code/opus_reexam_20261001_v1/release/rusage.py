"""TAG opus-release: per-process CPU counters from the macOS kernel (proc_pid_rusage, flavor 6), standard library only.

Why: on this shared machine the wall time of identical single-thread work swings 2-2.7x within seconds (measured 2026-10-01:
the performance cores are clocked down in bursts while other lanes and the GPU job run; the thread stays on performance cores).
INSTRUCTIONS of the work repeat to about +-0.5% between clean runs; CYCLES to about +-1% for matmul but up to +40% for streaming
adds under memory contention. Neither is proof against PAGING: when the machine is short of memory the kernel's fault handling
(decompressing pages) is counted in both, and a run reads 3-8% high. The tell-tale is system CPU: a clean run has system/user
0.13-0.21 in process and 0.20-0.26 for a server; above 0.30 the run paged and must be repeated.
Works on any pid of the same user (the client reads the server).
read(pid) -> dict(user_s, system_s, p_user_s, p_system_s, cycles, instructions, p_cycles, p_instructions, runnable_s,
                  footprint, footprint_max, resident, pageins)
footprint_max = lifetime peak of the physical footprint; unlike RSS it includes compressed pages, so it does not under-read when
the machine is short of memory."""
import ctypes
import os

_FIELDS = ['user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins', 'wired_size', 'resident_size',
           'phys_footprint', 'proc_start_abstime', 'proc_exit_abstime', 'child_user_time', 'child_system_time',
           'child_pkg_idle_wkups', 'child_interrupt_wkups', 'child_pageins', 'child_elapsed_abstime', 'diskio_bytesread',
           'diskio_byteswritten', 'cpu_time_qos_default', 'cpu_time_qos_maintenance', 'cpu_time_qos_background',
           'cpu_time_qos_utility', 'cpu_time_qos_legacy', 'cpu_time_qos_user_initiated', 'cpu_time_qos_user_interactive',
           'billed_system_time', 'serviced_system_time', 'logical_writes', 'lifetime_max_phys_footprint', 'instructions',
           'cycles', 'billed_energy', 'serviced_energy', 'interval_max_phys_footprint', 'runnable_time', 'flags', 'user_ptime',
           'system_ptime', 'pinstructions', 'pcycles', 'energy_nj', 'penergy_nj', 'secure_time_in_system',
           'secure_ptime_in_system', 'neural_footprint', 'lifetime_max_neural_footprint', 'interval_max_neural_footprint']


class _RusageV6(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8*16)] + [(n, ctypes.c_uint64) for n in _FIELDS] + [('reserved', ctypes.c_uint64*9)]


class _Timebase(ctypes.Structure):
    _fields_ = [('numer', ctypes.c_uint32), ('denom', ctypes.c_uint32)]


_proc = ctypes.CDLL('/usr/lib/libproc.dylib')
_tb = _Timebase()
ctypes.CDLL('/usr/lib/libSystem.B.dylib').mach_timebase_info(ctypes.byref(_tb))
_TICK = _tb.numer/_tb.denom*1e-9      # seconds per mach time unit
_buf = _RusageV6()
_ref = ctypes.byref(_buf)
_call = _proc.proc_pid_rusage
SELF = os.getpid()


def raw(pid=SELF):
    """(cycles, instructions) only: the cheap read used around every operation (about 1.7 microseconds)."""
    if _call(pid, 6, _ref):
        raise OSError(f'proc_pid_rusage failed for pid {pid}')
    return _buf.cycles, _buf.instructions


def read(pid=SELF):
    if _call(pid, 6, _ref):
        raise OSError(f'proc_pid_rusage failed for pid {pid}')
    b = _buf
    return dict(user_s=b.user_time*_TICK, system_s=b.system_time*_TICK, p_user_s=b.user_ptime*_TICK,
                p_system_s=b.system_ptime*_TICK, cycles=b.cycles, instructions=b.instructions, p_cycles=b.pcycles,
                p_instructions=b.pinstructions, runnable_s=b.runnable_time*_TICK, footprint=b.phys_footprint,
                footprint_max=b.lifetime_max_phys_footprint, resident=b.resident_size, pageins=b.pageins)


def delta(after, before):
    keep = ('footprint', 'footprint_max', 'resident')
    return {k: (after[k] if k in keep else after[k] - before[k]) for k in after}


if __name__ == '__main__':
    import json
    import sys
    print(json.dumps(read(int(sys.argv[1]) if len(sys.argv) > 1 else SELF), indent=1))
