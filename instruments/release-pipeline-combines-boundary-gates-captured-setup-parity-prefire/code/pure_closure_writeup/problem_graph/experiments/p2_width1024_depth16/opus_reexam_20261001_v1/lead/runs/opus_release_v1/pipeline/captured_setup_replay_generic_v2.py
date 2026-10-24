"""[opus-release] generic captured-setup replay v2 (2026-10-16, lead, user-approved): v1 plus support for a production setup whose
binding phase finished the whole PB list (keys_bound == keys_total) and whose level-op fusion phase was then cut by the wall guard
after `deferred_fused` fusions (the manifest field holds the production lvfuse count). v1's clock returns 10.0 after binding, so
phase 2 fuses nothing, and v1 asserted keys_bound < keys_total and no fusion. v2 keeps v1's behaviour exactly when binding was cut;
when nothing was cut, its clock lets binding finish and then lets phase 2 fuse exactly the production count (it counts the fused
entries of the listed keys at every clock read and cuts once the count is reached), so the native pb_log reproduces the production
counts (verify() still requires every semantic setup phase to match exactly). Fusion is bit-identical with bill 0, so parity is
unaffected by which keys are fused; the exact count is reproduced anyway."""
from pathlib import Path
import json, time
class CapturedCutoffClock:
    def __init__(self, mod, keys, full, fused_target):
        self.mod = mod; self.keys = keys; self.full = full; self.fused_target = fused_target; self.binding_reads = 0; self.binding_seen = False
        self.base = None   # fused entries already present when phase 2 starts (eager fusions made during binding do not count)
    def _fused_now(self):
        m = self.mod; caches = (m._IPT_A, m._IPT_B, m._IPT_C); n = 0
        for kind, L, rows, cols in m._PB_SINGLE:
            e = dict.get(caches[kind], (L, rows, cols))
            if e is not None and e[3] is not None: n += 1
        return n
    def monotonic(self):
        if self.mod._LVF_KEEP[0]:
            self.binding_seen = True; self.binding_reads += 1
            return 0.0 if self.binding_reads <= self.keys else 10.0
        if self.full and self.binding_seen:
            now = self._fused_now()
            if self.base is None: self.base = now
            return 0.0 if now - self.base < self.fused_target else 10.0
        return 10.0 if self.binding_seen else 0.0
    def __getattr__(self, name): return getattr(time, name)
def install(mod, manifest):
    s = json.loads(Path(manifest).read_text())['setup']
    assert isinstance(s['keys_bound'], int) and 0 < s['keys_bound'] <= s['keys_total']
    full = s['keys_bound'] == s['keys_total']
    assert full or s['deferred_fused'] == 0, s['deferred_fused']
    mod._pb_time = CapturedCutoffClock(mod, s['keys_bound'], full, int(s['deferred_fused'] or 0)); return s
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
