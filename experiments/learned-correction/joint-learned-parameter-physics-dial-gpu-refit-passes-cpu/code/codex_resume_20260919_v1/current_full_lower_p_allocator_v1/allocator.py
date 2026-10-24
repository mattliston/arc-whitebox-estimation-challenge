"""Documented all-zone unused-mapping relief; no direct live-memory release."""
import ctypes
import gc
import json
import os
import resource
import subprocess
import time
import weakref


class Stats(ctypes.Structure):
    _fields_ = [('blocks_in_use', ctypes.c_uint), ('size_in_use', ctypes.c_size_t),
                ('max_size_in_use', ctypes.c_size_t), ('size_allocated', ctypes.c_size_t)]


def abi():
    return dict(sizeof=ctypes.sizeof(Stats), offsets=[getattr(Stats, name).offset for name, _ in Stats._fields_])


class Allocator:
    def __init__(self):
        self.lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        self.lib.malloc_zone_statistics.argtypes = [ctypes.c_void_p, ctypes.POINTER(Stats)]
        self.lib.malloc_zone_statistics.restype = None
        self.lib.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.malloc_zone_pressure_relief.restype = ctypes.c_size_t

    def stats(self):
        result = Stats()
        self.lib.malloc_zone_statistics(None, ctypes.byref(result))
        return {name: getattr(result, name) for name, _ in Stats._fields_}

    def relieve(self):
        return self.lib.malloc_zone_pressure_relief(None, 0)


def rss():
    return int(subprocess.check_output(['/bin/ps', '-o', 'rss=', '-p', str(os.getpid())], text=True)) * 1024


class Recorder:
    def __init__(self, allocator, path):
        self.allocator = allocator
        self.stream = path.open('x')
        self.started = time.monotonic()
        self.count = 0
        self.call = None
        self.phase = None
        self.references = {}

    def track(self, state):
        if state is None:
            return
        for x in state:
            if hasattr(x, 'untyped_storage'):
                self.references[id(x)] = (weakref.ref(x), x.untyped_storage().data_ptr(), x.untyped_storage().nbytes())

    def boundary_bytes(self):
        self.references = {k: v for k, v in self.references.items() if v[0]() is not None}
        return sum(dict((v[1], v[2]) for v in self.references.values()).values())

    def point(self, event, layer=None, state=None):
        self.track(state)
        before = dict(malloc=self.allocator.stats(), rss_bytes=rss())
        collected = gc.collect()
        released = self.allocator.relieve()
        after = dict(malloc=self.allocator.stats(), rss_bytes=rss())
        result = dict(index=self.count, elapsed_s=time.monotonic() - self.started,
            call=self.call, phase=self.phase, event=event, layer=layer,
            before=before, after=after, gc_collected=collected,
            allocator_reported_released_bytes=released,
            weakly_tracked_boundary_storage_lower_bound_bytes=self.boundary_bytes(),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        self.stream.write(json.dumps(result, allow_nan=False) + '\n')
        self.stream.flush()
        self.count += 1
        return result


def wrapped_card(model, recorder):
    class AllocationCard(model.Card):
        def step(self, layer, state):
            recorder.point('step_enter', layer, state)
            result = None
            try:
                result = super().step(layer, state)
                return result
            finally:
                recorder.point('step_exit_or_checkpoint_early_stop', layer, result)
    return AllocationCard
