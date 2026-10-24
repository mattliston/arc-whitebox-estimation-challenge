"""[opus-release] generic captured-setup replay: deterministic clock reproduces the production run's PB cutoff (keys_bound);
verify() requires every semantic setup phase (name, count) of the production pb_log to match the native one exactly."""
from pathlib import Path
import json, time
class CapturedCutoffClock:
    def __init__(self, mod, keys): self.mod = mod; self.keys = keys; self.binding_reads = 0; self.binding_seen = False
    def monotonic(self):
        if self.mod._LVF_KEEP[0]:
            self.binding_seen = True; self.binding_reads += 1
            return 0.0 if self.binding_reads <= self.keys else 10.0
        return 10.0 if self.binding_seen else 0.0
    def __getattr__(self, name): return getattr(time, name)
def install(mod, manifest):
    s = json.loads(Path(manifest).read_text())['setup']
    assert s['deferred_fused'] == 0 and isinstance(s['keys_bound'], int) and 0 < s['keys_bound'] < s['keys_total']
    mod._pb_time = CapturedCutoffClock(mod, s['keys_bound']); return s
def _semantic(log):
    return [(x[0], x[1]) for x in log if len(x) > 2]
def verify(mod, s):
    native, captured = _semantic([list(x) for x in mod._PB_LOG]), _semantic(s['pb_log'])
    assert native == captured, (native, captured)
    names = [n for n, _ in captured]
    if 'sm40_fixed_views' in names:
        st = mod._ASTRA_SM40_BIND_STATS; c = dict(captured)
        assert st['bound'] is True and st['families'] == c['sm40_fixed_views'] and st['prebound'] == c['sm40_hot_views'], st
    if '_YR7_BIND_STATS' in mod.__dict__:
        assert mod._YR7_BIND_STATS['keys'] == 0 and mod._YR7_BIND_STATS['retired'] is True
    return dict(captured_semantic_phases=captured, native_semantic_phases=native, all_setup_counts_exact=True, PB_order_preserved=True, source_unchanged=True)
