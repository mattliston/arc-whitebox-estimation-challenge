#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-SETUP): appends a setup(context) hook that binds NETWORK-INDEPENDENT state once per worker, off-budget,
before any predict() (the estimator contract: setup runs outside the MLP's BudgetContext; whestbench times it against
setup_timeout_s ~5 s including the import). Stackable last on any built candidate (no in-place edits).
What it binds, in order, each step only while inside the wall budget (default 2.0 s measured inside setup):
  1. the kernel arena and its layout markers (the live loop's own init block: same code, so identical state);
  2. the fixed-shape kernel products on zero operands, which bind exactly the frames the predictions look up (keys are shapes and
     arena views only): batched 1024^3 D5 stages P=2 and P=3 (if --batch), a single 1024^3 square (final-layer diagonals) and the
     layer-0 gram (three 512x1024x512 products). Values written to the arena are scratch;
  3. the body's parameter uploads and buffers (GNN, message edge, birth head, bd coefficients, pair / bd / edge / feature state,
     pair-map arena views, per-layer GNN parameter views).
Session: if no flopscope session is active (NoBudgetContextError on the first op), the hook opens its own BudgetContext.
Any other exception stops the hook silently: predict() binds lazily exactly as before (every binder stores whole entries only).
usage: build_e2setup.py <in.py> <out.py> [budget_s=2.0]"""
import sys
from pathlib import Path
FRAGMENT = '''

# E2-SETUP (engineer2, build_e2setup.py): bind network-independent state in setup(), off-budget, once per worker.
_E2_SETUP_BUDGET_S = {budget}
_E2_SETUP_LOG = []
def _e2_setup_steps(deadline):
    import time as _time
    global _SCRATCH, _GNN_PARAMS, _EDGE_PARAMETERS
    def left():
        return _time.time() < deadline
    if _SCRATCH is None:                                        # 1. the live loop's own arena init
        arena = fnp.empty((_PACKED_ELEMENTS,), dtype=fnp.float32)
        _SCRATCH = [arena, _cp3_bind_arena_views(arena, _PACKED_PUSH), _cp3_bind_arena_views(arena, _PACKED_READ), None, None, None, None]
        _bind_push6(_SCRATCH[0])
        _E2_SETUP_LOG.append('arena')
    z = fnp.zeros((1024, 1024), dtype=fnp.float32)              # 2. fixed-shape products (frames keyed by shape + arena views)
    if left() and '_mm_batch' in globals():
        _mm_batch([(z, z), (z, z)]); _E2_SETUP_LOG.append('batch2')
    if left() and '_mm_batch' in globals():
        _mm_batch([(z, z), (z, z), (z, z)]); _E2_SETUP_LOG.append('batch3')
    if left():
        _mm(z, z); _E2_SETUP_LOG.append('square')
    if left():
        _resident_gram512(z.T, z); _E2_SETUP_LOG.append('gram512')
    if left():                                                  # 3. body parameters and buffers (all keyed by constants)
        if _GNN_PARAMS is None:
            _GNN_PARAMS = _unpack_static_fp32(_GNN_LITERAL)
        _message_edge_parameters(); _bdnh_parameters(); _bd_coefficients()
        if _EDGE_PARAMETERS is None:
            _EDGE_PARAMETERS = {{k: fnp.asarray(v, dtype=fnp.float32) for k, v in _EDGE_LITERAL.items()}}
        _E2_SETUP_LOG.append('params')
    if left():
        k = _e2_gnn_consts(_GNN_PARAMS, 1024)
        for l in range(1, 16):
            _e2_gnn_layer(k, l)
        _e2_edge_consts(_EDGE_PARAMETERS, 1024); _e2_pair_state(1024); _e2_bd_state(1024)
        for l in range(16):
            _e2_feature_columns(1024, l)
        if '_e2_pair_arena' in globals():
            _e2_pair_arena(1024)
        _E2_SETUP_LOG.append('state')
def _e2_setup(self, context):
    import time as _time
    deadline = _time.time() + _E2_SETUP_BUDGET_S
    try:
        try:
            _e2_setup_steps(deadline)
        except flops.NoBudgetContextError:
            with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace='e2-setup'):
                _e2_setup_steps(deadline)
    except Exception as exc:                                    # never fail setup: predict binds lazily as before
        _E2_SETUP_LOG.append('stopped: ' + type(exc).__name__)
Estimator.setup = _e2_setup
'''
if __name__ == "__main__":
    src = Path(sys.argv[1]).read_text()
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    for need in ("\nclass Estimator(BaseEstimator):\n", "\ndef _resident_gram512(", "\ndef _e2_gnn_consts(", "\ndef _unpack_static_fp32(",
                 "\n_GNN_PARAMS = None", "\n_EDGE_PARAMETERS=None", "_bind_push6(_SCRATCH[0])"):
        assert need in src, f"base lacks {need.strip()!r}"
    assert "_E2_SETUP_BUDGET_S" not in src
    out = Path(sys.argv[2]); assert not out.exists(), f"refusing to overwrite {out}"
    out.write_text(src.rstrip("\n") + "\n" + FRAGMENT.format(budget=budget))
    print(f"E2-SETUP budget {budget}s: {sys.argv[1]} -> {sys.argv[2]}")
