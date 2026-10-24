#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-MEM): bit-exact LIVE-PEAK diet for the live _predict_binary loop, stackable on any built candidate
(S2/S3/S4 chain; e2 v7 or v8 body). The peak locator (server_mem_v2.py) puts the S4 server live-array peak (3,015 MB r35,
3,010 MB r100) at the diagonal line of the live loop, where four (n, K) float64 arrays (178 MB each at K ~ 22.8k), the dead
previous factors p and the dead read operands left/right are all alive next to the 1,494 MB kernel arena. Edits:
  1. diagonal: (pp2*pp64) then *lam64 IN PLACE (same two float64 products per element, same order; the ordered row sum
     runs on the same C-contiguous values) -> one (n, K) float64 transient fewer.
  2. del left, right right after the old-tier read (they are dead there; they lived to the next layer).
  3. del p right after the push (dead until the end-of-layer reassignment).
No arithmetic changes; no bill change. Requires the T31 locals() plumbing to be off (it could read the deleted names).
Optional stage `yright` (bit-exact, bill -0.1%): the live loop gathers right = pp[live_indices, old_columns:] only (the old-tier read
never touches the old columns of right; they reach the plane through the frame), read by _e2_shared_old384_read_young, a copy of
the live _shared_old384_read that takes the young columns directly (same young_right values -> same fresh product -> same _mm input).
Edit 1 also accepts engineer's float32 third-order form (build_f32.py `third`, E-29): the same in-place *lam on its float32
product (bit-exact relative to that form). Exactly one of the two diagonal forms must be present.
usage: build_e2mem.py <in_cand.py> <out_cand.py> [stages, default mem; e.g. mem,yright]
       (v1 = build_e2mem_v1_3edits.py; v2 = build_e2mem_v2_yright.py, float64 diagonal form only; v3 = build_e2mem_v3_f32form.py,
       e2 v7 read line only). v4 also accepts the e2 v8 (hold2) read line `_e2_H(_shared_old384_read(...)).astype(...)`.
Optional stage `arena` (v5, bit-exact, bill-identical): the pair map's three series products ((6,3,n,n), (5,3,3,n,n), (4,2,3,n,n)
float32, 72+180+96 MB) and its two tri sums ((5,3,n,n), (4,3,n,n)) are written with out= into views of the kernel arena, which is
pure scratch between kernel calls (engineer, 02:03) and the pair map makes none: no fresh allocation (zero-fill) of ~456 MB per layer.
Views bound once per process (reshape billed once); the sums keep their exact C-contiguous shapes, so every value is identical.
Optional stage `hold2` (v6, bit-exact, bill-identical; run it ALONE on a finished file, e.g. `build_e2mem.py S7_mem.py out hold2`, since
it must be the very last edit): the e2 v8 AST pass (hold2.py) applied to the body functions of the final stack, including the porter's
_gn_dcov and _k31_row_scale: every temporary-producing operand is kept in a per-call (per-iteration in the live loop) _E2Hold list, so
the client frees a call's temporaries in one hidden free trip instead of one per statement (fewer round trips -> less residual)."""
import sys
from pathlib import Path
LIVE_DEF = "def _predict_binary(mlp, hook=None, cov32=True, metric='covariance', collect_states=False, gate_floor=0.01, final_hook=None, coefficient_hook=None, covariance_hook=None):\n"
LIVE_END = "\n\n# Fixed Weight5 lazy-rotation port."
EDITS = [
    ("                diagonal = fnp.sum(_e2_pp2 * _e2_pp64 * _e2_lam64, axis=1)\n"
     "                del _e2_pp64, _e2_pp2\n",
     "                _e2_t3 = _e2_pp2 * _e2_pp64\n"
     "                _e2_t3h = fnp.multiply(_e2_t3, _e2_lam64, out=_e2_t3)   # E2-M: in place (same products and order): one (n,K) float64 transient fewer at the live peak\n"
     "                diagonal = fnp.sum(_e2_t3, axis=1)\n"
     "                del _e2_pp64, _e2_pp2, _e2_t3, _e2_t3h\n"),
    ("                response_plane = _shared_old384_read(left, right, lam, live_indices, shared_old).astype(fnp.float64)\n",
     "                response_plane = _shared_old384_read(left, right, lam, live_indices, shared_old).astype(fnp.float64)\n"
     "                del left, right   # E2-M: dead after the read (they lived to the next layer, through the live peak)\n"),
    ("\n            pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)\n",
     "\n            pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)\n"
     "            del p   # E2-M: dead until the end-of-layer reassignment (it lived through the live peak)\n"),
]
YRIGHT_EDITS = [
    ("                right = pp[live_indices]\n"
     "                response_plane = _shared_old384_read(left, right, lam, live_indices, shared_old).astype(fnp.float64)\n",
     "                right = pp[live_indices, int(shared_old['columns']):]   # E2-R: only the young columns of right are ever read\n"
     "                response_plane = _e2_shared_old384_read_young(left, right, lam, live_indices, shared_old).astype(fnp.float64)\n"),
]
READ_LIVE = """def _shared_old384_read(left, right, lam, live_indices, state):
    old_columns = int(state['columns'])
    if not old_columns:
        return _mm(left * left, (right * lam).T)
    old_left = left[:, :old_columns]
    inner = _slr_inner(old_left, lam[:old_columns], state)
    old_plane = _shared_old384_mm(inner, state['pre_basis'][live_indices, :].T)
    young_left, young_right = left[:, old_columns:], right[:, old_columns:]
    young_plane = _mm(young_left * young_left, (young_right * lam[old_columns:]).T)
    return young_plane + old_plane
"""
READ_YOUNG = """

# E2-R (engineer2, build_e2mem.py stage yright): the live _shared_old384_read with right = pp[live, old_columns:] passed directly.
def _e2_shared_old384_read_young(left, young_right, lam, live_indices, state):
    old_columns = int(state['columns'])
    if not old_columns:
        return _mm(left * left, (young_right * lam).T)
    old_left = left[:, :old_columns]
    inner = _slr_inner(old_left, lam[:old_columns], state)
    old_plane = _shared_old384_mm(inner, state['pre_basis'][live_indices, :].T)
    young_left = left[:, old_columns:]
    young_plane = _mm(young_left * young_left, (young_right * lam[old_columns:]).T)
    return young_plane + old_plane
"""
DIAG_F32 = ("                diagonal = fnp.sum(_e2_pp2 * pp * lam, axis=1).astype(fnp.float64)\n"
            "                del _e2_pp2\n",
            "                _e2_t3 = _e2_pp2 * pp\n"
            "                _e2_t3h = fnp.multiply(_e2_t3, lam, out=_e2_t3)   # E2-M: in place (same products and order): one (n,K) transient fewer at the live peak\n"
            "                diagonal = fnp.sum(_e2_t3, axis=1).astype(fnp.float64)\n"
            "                del _e2_pp2, _e2_t3, _e2_t3h\n")
READ_V7 = "_shared_old384_read(left, right, lam, live_indices, shared_old).astype(fnp.float64)\n"
READ_V8 = "_e2_H(_shared_old384_read(left, right, lam, live_indices, shared_old)).astype(fnp.float64)\n"
def _v8_form(old, new):
    """the same edit on the e2 v8 (hold2) spelling of the old-tier read line"""
    return old.replace(READ_V7, READ_V8), new.replace(READ_V7, READ_V8).replace(
        "_e2_shared_old384_read_young(left, right, lam, live_indices, shared_old).astype(fnp.float64)\n",
        "_e2_H(_e2_shared_old384_read_young(left, right, lam, live_indices, shared_old)).astype(fnp.float64)\n")
def _live_edit(src, edits):
    i = src.index(LIVE_DEF); j = src.index(LIVE_END, i)
    live = src[i:j]
    for old, new in edits:
        if READ_V7 in old and live.count(old) == 0:
            old, new = _v8_form(old, new)   # e2 v8 (hold2) body
        if old == EDITS[0][0] and live.count(old) == 0 and live.count(DIAG_F32[0]) == 1:
            old, new = DIAG_F32          # engineer's float32 third-order form (build_f32.py third)
        assert live.count(old) == 1, f"expected 1x in the live loop, found {live.count(old)}x: {old.strip()[:70]!r}"
        live = live.replace(old, new)
    return src[:i] + live + src[j:]
ARENA_EDITS = [
    ("    X = st['Fr16'] * st['Gr16']\n"
     "    fnp.multiply(X, st['P6'], out=X)\n",
     "    _e2_A = _e2_pair_arena(n)   # E2-A: the series products and tri sums live in the idle kernel arena (no fresh allocation)\n"
     "    X = _e2_A['X1']\n"
     "    _e2_hold.append(fnp.multiply(st['Fr16'], st['Gr16'], out=X))\n"
     "    _e2_hold.append(fnp.multiply(X, st['P6'], out=X))\n"),
    ("    X = st['Fr'][[[2, 3, 4], [1, 2, 3], [2, 3, 4], [3, 4, 5], [1, 2, 3]]] * st['Gr'][[[1, 2, 3], [2, 3, 4], [2, 3, 4], [1, 2, 3], [3, 4, 5]]]\n"
     "    fnp.multiply(X, st['RR'], out=X)\n"
     "    T1, T2, T3, T4, T5 = fnp.unstack(fnp.sum(X, axis=1))",
     "    X = _e2_A['X2']\n"
     "    _e2_hold.append(fnp.multiply(st['Fr'][[[2, 3, 4], [1, 2, 3], [2, 3, 4], [3, 4, 5], [1, 2, 3]]], st['Gr'][[[1, 2, 3], [2, 3, 4], [2, 3, 4], [1, 2, 3], [3, 4, 5]]], out=X))\n"
     "    _e2_hold.append(fnp.multiply(X, st['RR'], out=X))\n"
     "    T1, T2, T3, T4, T5 = fnp.unstack(fnp.sum(X, axis=1, out=_e2_A['T']))"),
    ("    X = (st['F5'][[[1, 1], [0, 0], [2, 2], [0, 0]], [[4, 5], [1, 2], [5, 6], [1, 2]]]\n"
     "         * st['G5'][[[0, 0], [1, 1], [0, 0], [2, 2]], [[1, 2], [4, 5], [1, 2], [5, 6]]])\n"
     "    fnp.multiply(X, st['P12'], out=X)\n"
     "    T6, T7, T8, T9 = fnp.unstack(fnp.sum(X, axis=1))",
     "    X = _e2_A['X3']\n"
     "    _e2_hold.append(fnp.multiply(st['F5'][[[1, 1], [0, 0], [2, 2], [0, 0]], [[4, 5], [1, 2], [5, 6], [1, 2]]],\n"
     "                                 st['G5'][[[0, 0], [1, 1], [0, 0], [2, 2]], [[1, 2], [4, 5], [1, 2], [5, 6]]], out=X))\n"
     "    _e2_hold.append(fnp.multiply(X, st['P12'], out=X))\n"
     "    T6, T7, T8, T9 = fnp.unstack(fnp.sum(X, axis=1, out=_e2_A['U']))"),
]
ARENA_FRAGMENT = """

# E2-A (engineer2, build_e2mem.py stage arena): pair-map transients as views of the kernel arena (pure scratch between kernel calls;
# the pair map makes none). X1/X2/X3 share the region [0, 45 n^2) (used one after another), T at [45, 60) n^2, U at [60, 72) n^2.
_E2_PAIR_ARENA = {}
def _e2_pair_arena(n):
    arena = _SCRATCH[0]
    got = _E2_PAIR_ARENA.get(n)
    if got is not None and got[0] is arena:
        return got[1]
    nn = n * n
    assert int(arena.shape[0]) >= 72 * nn, 'kernel arena smaller than the pair-map scratch'
    views = dict(X1=fnp.reshape(arena[0:18 * nn], (6, 3, n, n)), X2=fnp.reshape(arena[0:45 * nn], (5, 3, 3, n, n)),
                 X3=fnp.reshape(arena[0:24 * nn], (4, 2, 3, n, n)), T=fnp.reshape(arena[45 * nn:60 * nn], (5, 3, n, n)),
                 U=fnp.reshape(arena[60 * nn:72 * nn], (4, 3, n, n)))
    _E2_PAIR_ARENA[n] = (arena, views)
    return views
"""
PAIR_DEF = "def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity', extra31=None):\n"
HOLD2_LAST = ["def gaussian_raw(mu, v):\n", "def derivatives(a, pdf, g, mg_over_s, order):\n", "def features(l, m, C, K21, aux, W):\n",
              "def silu(x):\n", "def gnn_step(l,x,sigma,W,state,P):\n", PAIR_DEF, "def nonlinear_scalar(mu, v, k3, k4):\n",
              "def final_scalar_features(l,m,aux,W):\n",   # _nonlinear_first_gaussian excluded: holding its r-loop float64 planes adds ~200 MB at layer 0
              "def omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host):\n",
              "def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):\n", "def _e2_bdnh_coefficients(H, A, cstar_all, l):\n",
              "def _shared_old384_finite(value, name):\n", "def _shared_old384_push(layer, w32, p, previous, newborn, lam, state):\n",
              "def _shared_old384_read(left, right, lam, live_indices, state):\n", "def _shared_old384_commit(layer, gate, alive, selected, state):\n",
              "def _slr_inner(old_left, lam_old, state):\n", "def _slr_join_factors(layer, overlap, state, Fj):\n", "def _slr_reconstruct(U, state):\n",
              "def _slr_virtual_rotation_envelope(overlap, base):\n", "def _push_bank_shared_newborn(w,p,previous,newborn):\n",
              "def _flms_star(w, A):\n", "def _flms_covariance(w, C):\n", "def _flms_star_diagonal(w, A):\n", "def _flms_covariance_diagonal(w, C):\n",
              "def _message_edge_source(layer,m,K21,aux,w,state,P,context):\n", "def _e2_wsums(st):\n", "def _e2_messages(k, W, state):\n",
              "def _e2_wpow(W, need6=False):\n", "def _e2_ms(m, s):\n", "def _cp_diagonal_cube(x):\n",
              "def _gn_dcov(l, mu, Cpre, K21, k4, aux):\n", "def _k31_row_scale(mu, k3, v):\n"]
HOLD2_LAST_SKIP = ("            lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])",
                   "    common = _w5_port_call(_W5_PORT_SCOPE,w,previous,'newborn_common',_mm,wp,base[previous,:])",
                   "            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)")
def _hold2_last(src):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent)); import hold2
    assert "class _E2Hold(list):" not in src, "hold2 already applied"
    assert src.count("\n_E2_LAYER = {}") == 1
    src = src.replace("\n_E2_LAYER = {}", "\n" + hold2.HOLD_CLASS.strip() + "\n\n\n_E2_LAYER = {}", 1)
    total = 0
    for start in HOLD2_LAST:
        assert start in src, f"hold2 target missing: {start.strip()[:70]!r}"
        src, k = hold2.apply(src, start, HOLD2_LAST_SKIP, start == LIVE_DEF)
        total += k
    src, k = hold2.apply(src, LIVE_DEF, HOLD2_LAST_SKIP, True)
    total += k
    print(f"hold2 (last): {total} statements rewritten")
    return src
def build(src, stages=("mem",)):
    assert src.count(LIVE_DEF) == 1, "live _predict_binary signature not unique"
    for guard in ("\n_T31_LAYERS = ()\n", "\n_t31_commit = None\n"):
        assert src.count(guard) == 1, f"T31 locals() plumbing must be off: {guard.strip()!r}"
    if "mem" in stages:
        src = _live_edit(src, EDITS)
    if "yright" in stages:
        # the LAST definition of _shared_old384_read is the live one and must be exactly the form the young reader copies
        k = src.rindex("def _shared_old384_read(")
        assert src[k:].startswith(READ_LIVE), "live _shared_old384_read differs from the form _e2_shared_old384_read_young copies"
        src = _live_edit(src, YRIGHT_EDITS)
        src = src.rstrip("\n") + "\n" + READ_YOUNG
    if "arena" in stages:
        assert src.count(PAIR_DEF) == 1, "live pair map signature not unique"
        i = src.index(PAIR_DEF); j = src.index("\nnonlinear=_nonlinear_pair_batch\n", i)
        body = src[i:j]
        for old, new in ARENA_EDITS:
            assert body.count(old) == 1, f"pair map: expected 1x, found {body.count(old)}x: {old.strip()[:60]!r}"
            body = body.replace(old, new)
        src = src[:i] + body + src[j:]
        src = src.rstrip("\n") + "\n" + ARENA_FRAGMENT
    if "hold2" in stages:
        src = _hold2_last(src)
    return src
if __name__ == "__main__":
    src = Path(sys.argv[1]).read_text()
    stages = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("mem",)
    out = build(src, stages)
    Path(sys.argv[2]).write_text(out)
    print(f"E2-MEM {'+'.join(stages)}: {sys.argv[1]} -> {sys.argv[2]} (+{out.count(chr(10)) - src.count(chr(10))} lines)")
