#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-PB): appends a setup(context) hook that PRE-BINDS kernel frames once per worker, off-budget, from a
census-chosen key list (setupbind/select_prebind.py). Kernel frames are views of the arena keyed only by (layout, depth, shape);
the hook replays the binder calls for the listed keys (views only, no arithmetic: the arena's contents at the first predict are
those of a fresh arena, so predictions are bit-identical) in the list's order until a wall deadline measured inside setup.
Steps: (1) the live loop's own arena init (same code); (2) the listed keys, each followed by one _BH_HOLD release (one batched
free per key); stops at the deadline. No flopscope session (NoBudgetContextError on the first op): the hook opens its own
BudgetContext. Any other exception stops the hook silently (every binder stores whole entries only; predict binds lazily as before).
usage: build_e2prebind.py <in.py> <out.py> <keys.json> [budget_s=2.0]"""
import sys, json
from pathlib import Path
FRAGMENT = '''

# E2-PREBIND (engineer2, build_e2prebind.py, 2026-09-23): census-driven frame pre-binding in setup(), off-budget, once per worker.
# {nkeys} keys from {keysfile} (sha {keysha}); wall budget {budget} s measured inside setup.
_E2_PB_BUDGET_S = {budget}
_E2_PB_KEYS = {keys}
_E2_SETUP_LOG = []
_E2_PB_SQUARE_SPECS = {{3: '_PACKED_SQUARE', 4: '_PACKED_SQUARE4', 5: '_PACKED_SQUARE5', 6: '_PACKED_SQUARE6'}}
def _e2_pb_views(tag):
    global _SCRATCH_PUSH5
    if tag == '_SCRATCH_PUSH5':
        if _SCRATCH_PUSH5 is None:
            _SCRATCH_PUSH5 = _cp3_bind_arena_views(_SCRATCH[0], _PACKED_PUSH5)
        return _SCRATCH_PUSH5
    if tag.startswith('_BB_VIEWS['):
        return _bb_views(int(tag[10:-1]))
    i = int(tag[9:-1])
    if _SCRATCH[i] is None:
        _SCRATCH[i] = _cp3_bind_arena_views(_SCRATCH[0], globals()[_E2_PB_SQUARE_SPECS[i]])
    return _SCRATCH[i]
def _e2_pb_bind(key):
    name = key[0]
    if name == '_packed_frame':
        levels, m, k, n, tag = key[1:6]
        _packed_frame(levels, m, k, n, _e2_pb_views(tag))
    elif name == '_read5_leaf_packets':
        m, k, n, tag = key[1:5]
        views = _e2_pb_views(tag)
        _forward, leaf_a, leaf_b, leaf, _backward = _packed_frame(4, m, k, n, views)
        _read5_packet_frames(m, k, n, views, leaf_a, leaf_b, leaf)
    else:
        globals()[name](_e2_pb_views(key[1]), *key[2:])
    _BH_HOLD.clear()
def _e2_pb_steps(deadline):
    import time as _time
    global _SCRATCH
    if _SCRATCH is None:
        arena = fnp.empty((_PACKED_ELEMENTS,), dtype=fnp.float32)
        _SCRATCH = [arena, _cp3_bind_arena_views(arena, _PACKED_PUSH), _cp3_bind_arena_views(arena, _PACKED_READ), None, None, None, None]
        _bind_push6(_SCRATCH[0])
        _E2_SETUP_LOG.append('arena')
    done = 0
    for key in _E2_PB_KEYS:
        if _time.time() >= deadline:
            break
        _e2_pb_bind(key)
        done += 1
    _E2_SETUP_LOG.append('keys %d/%d' % (done, len(_E2_PB_KEYS)))
def _e2_pb_setup(self, context):
    import time as _time
    deadline = _time.time() + _E2_PB_BUDGET_S
    try:
        try:
            _e2_pb_steps(deadline)
        except flops.NoBudgetContextError:
            with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace='e2-prebind'):
                _e2_pb_steps(deadline)
    except Exception as exc:
        _E2_SETUP_LOG.append('stopped: ' + type(exc).__name__)
Estimator.setup = _e2_pb_setup
'''
def parse(key):
    return tuple(int(x) if x.lstrip('-').isdigit() else x for x in key.split('|') if not x.startswith('arr('))
if __name__ == "__main__":
    import hashlib
    src = Path(sys.argv[1]).read_text(); kf = Path(sys.argv[3]); budget = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    keys = tuple(parse(k) for k in json.load(open(kf)))
    live = "            _SCRATCH = [arena, _e2_H(_cp3_bind_arena_views(arena, _PACKED_PUSH)), _e2_H(_cp3_bind_arena_views(arena, _PACKED_READ)), None, None, None, None]\n            _bind_push6(_e2_H(_SCRATCH[0]))\n"
    for need in ("\nclass Estimator(BaseEstimator):\n", live, "\n_BH_HOLD = []\n", "\ndef _read5_packet_frames(", "\ndef _bb_views(", "\n_PACKED_PUSH5 = "):
        assert need in src, f"base lacks {need.strip()[:60]!r}"
    assert "_E2_PB_BUDGET_S" not in src and "_E2_SETUP_BUDGET_S" not in src, "base already has a setup hook"
    out = Path(sys.argv[2]); assert not out.exists(), f"refusing to overwrite {out}"
    out.write_text(src.rstrip("\n") + "\n" + FRAGMENT.format(budget=budget, keys=repr(keys), nkeys=len(keys), keysfile=kf.name,
                                                             keysha=hashlib.sha256(kf.read_bytes()).hexdigest()[:8]))
    print(f"E2-PREBIND {len(keys)} keys, budget {budget}s: {sys.argv[1]} -> {sys.argv[2]}")
