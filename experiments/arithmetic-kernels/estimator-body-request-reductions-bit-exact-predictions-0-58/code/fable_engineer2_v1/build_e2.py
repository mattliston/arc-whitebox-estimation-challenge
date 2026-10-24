#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-B): request-diet builder for the card body OUTSIDE the Winograd kernel. Applies exact string edits (each with
a count assert, so it stacks with other patches and fails loudly on a changed base) to <base_cand.py> and writes <out.py>.
usage: build_e2.py <base_cand.py> <out.py> [patch,patch,...]   (default: all patches, in order)"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent

def between(src, start, end):
    """Exact text of the block that begins with `start` and ends just before `end` (both must occur once)."""
    assert src.count(start) == 1, f"block start occurs {src.count(start)}x: {start[:60]!r}"
    i = src.index(start); j = src.index(end, i)
    assert src.count(end) >= 1
    return src[i:j]

def patch_pair(src):
    old = between(src, "def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity'):\n", "\nnonlinear=_nonlinear_pair_batch\n")
    new = (HERE / "patches/pair_new.py").read_text()
    assert src.count(old) == 1
    return src.replace(old, new + "\n")

def patch_bd(src):
    old = between(src, "def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):\n", "\n# Frozen node-conditioned birth-column prototype")
    new = (HERE / "patches/bd_new.py").read_text()
    assert src.count(old) == 1
    return src.replace(old, new + "\n")

def replace_block(src, start, end, patch_file, tail=""):
    old = between(src, start, end) if end is not None else src[src.index(start):]
    assert src.count(start) == 1 and src.count(old) == 1
    return src.replace(old, (HERE / patch_file).read_text() + tail)

def replace_exact(src, old, new, count=1):
    assert src.count(old) == count, f"expected {count}x, found {src.count(old)}x: {old[:80]!r}"
    return src.replace(old, new)

def patch_learned(src):
    """features / gnn_step / final_scalar_features / _message_edge_source: shared per-layer weight powers, batched message
    products (shared by the edge source and gnn_step), cached parameter views; the hook skips the unused layer-0 features."""
    src = replace_block(src, "def features(l, m, C, K21, aux, W):\n", "\n_SCRATCH = None\n", "patches/features_new.py")
    src = replace_block(src, "def gnn_step(l,x,sigma,W,state,P):\n", "\n_GNN_LITERAL = {", "patches/gnn_new.py", "\n")
    src = replace_block(src, "def final_scalar_features(l,m,aux,W):\n", "def final_input(x, W, state, P):\n", "patches/final_features_new.py")
    src = replace_block(src, "def _message_edge_source(layer,m,K21,aux,w,state,P,context):\n", None, "patches/edge_new.py")
    src = replace_exact(src, "        x=features(l,m,C,K21,aux,w)\n        delta,state=gnn_step(l,x,aux[\"sigma\"],w,state,P)\n        if 1 <= l <= 14:\n",
                        "        x=features(l,m,C,K21,aux,w) if l >= 1 else None   # E2: gnn_step ignores x at layer 0\n        delta,state=gnn_step(l,x,aux[\"sigma\"],w,state,P)\n        if 1 <= l <= 14:\n")
    return src

LIVE_DEF = "def _predict_binary(mlp, hook=None, cov32=True, metric='covariance', collect_states=False, gate_floor=0.01, final_hook=None, coefficient_hook=None, covariance_hook=None):\n"

def patch_loop(src):
    """Live-loop request diet (+ its consumers): shared gate chain / outer(v,v) / Cpre*Cpre / float64 pp copies / beta, outer()
    instead of broadcast views, no redundant copies; requires the pair and bd patches (they read _E2_LAYER)."""
    assert "_E2_LAYER = {}" in src, "patch_loop needs patch_pair first"
    live = between(src, LIVE_DEF, "\n\n# Fixed Weight5 lazy-rotation port.")
    ns = {}; exec((HERE / "patches/loop_edits.py").read_text(), ns)
    new_live = live
    for old, new in ns["EDITS"]:
        new_live = replace_exact(new_live, old, new)
    src = replace_exact(src, live, new_live)
    live2 = between(src, "def _push_bank_shared_newborn(w,p,previous,newborn):\n    wp = w[previous,:].T\n    if newborn is None:\n        return _mm(wp,p[previous,:])\n    old_columns,selected,base,t,s = newborn\n    inherited = _mm(wp,p[previous,:old_columns])\n    common = _w5_port_call(", "\ndef _shared_old384_push(")
    src = replace_exact(src, live2, replace_exact(live2, "    return fnp.concatenate([inherited,common+directions*t[None,:],\n                            common+directions*s[None,:]],axis=1)\n",
                                                  "    return fnp.concatenate([inherited,common+directions*t,\n                            common+directions*s],axis=1)\n"))
    src = replace_block(src, "def omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host):\n", "def _nonlinear_terminal_live(", "patches/omitted_new.py")
    src = replace_exact(src, "        Q = gate.astype(fnp.float32)[:, None] * state['pre_basis']\n        state['Q'] = alive.astype(fnp.float32)[:, None] * Q\n",
                        "        L = _E2_LAYER   # E2: alive_i*(g_i*x) == (g_i*alive_i)*x exactly; the live loop already formed that column\n"
                        "        ga = L['ga'] if L.get('ga_gate') is gate and L.get('ga_alive') is alive else (gate.astype(fnp.float32) * alive)[:, None]\n"
                        "        state['Q'] = ga * state['pre_basis']\n")
    src = replace_exact(src, "    wt = w.T\n    pushed = _mm(wt, A.astype(fnp.float32))\n    sub = (2*_mm(pushed*wt, w) + _mm(wt*wt, pushed.T))/3\n    diagonal = fnp.sum(pushed*wt*wt, axis=1, dtype=fnp.float64)\n",
                        "    wt = w.T\n    pushed = _mm(wt, A if str(A.dtype) == 'float32' else A.astype(fnp.float32))\n    pw = pushed*wt\n    sub = (2*_mm(pw, w) + _mm(wt*wt, pushed.T))/3\n    diagonal = fnp.sum(pw*wt, axis=1, dtype=fnp.float64)\n")
    return src

def patch_fuse(src):
    """einsum(optimize=False) products instead of broadcast views (+ gaussian_raw CSE); requires pair, bd, learned, loop."""
    ns = {}; exec((HERE / "patches/fuse_edits.py").read_text(), ns)
    live = between(src, LIVE_DEF, "\n\n# Fixed Weight5 lazy-rotation port.")
    new_live = live
    for old, new in ns["LIVE"]:
        new_live = replace_exact(new_live, old, new)
    src = replace_exact(src, live, new_live)
    for old, new in ns["OTHER"]:
        src = replace_exact(src, old, new)
    return src

def patch_vec(src):
    """batched per-unit vector algebra in the pair map, shared (a, t*alive, s*alive) float32 gathers, cached birth-head rows,
    boolean/outer simplifications in the live loop; requires pair, bd, learned, loop, fuse."""
    ns = {}; exec((HERE / "patches/vec_edits.py").read_text(), ns)
    for old, new in ns["OTHER"]:
        src = replace_exact(src, old, new)
    live = between(src, LIVE_DEF, "\n\n# Fixed Weight5 lazy-rotation port.")
    new_live = live
    for old, new in ns["LIVE"]:
        new_live = replace_exact(new_live, old, new)
    return replace_exact(src, live, new_live)

def patch_share(src):
    """pair-map outer products in one persistent float64 stack (one promote-and-scale multiply), s s^T / mg/s / m/s shared with
    features and the message edge source; requires pair, bd, learned, loop, fuse, vec."""
    ns = {}; exec((HERE / "patches/share_edits.py").read_text(), ns)
    for old, new in ns["OTHER"]:
        src = replace_exact(src, old, new)
    edge = src[src.index("def _message_edge_source(layer,m,K21,aux,w,state,P,context):\n"):]
    new_edge = edge
    for old, new in ns["EDGE"]:
        new_edge = replace_exact(new_edge, old, new)
    src = replace_exact(src, edge, new_edge)
    live = between(src, LIVE_DEF, "\n\n# Fixed Weight5 lazy-rotation port.")
    new_live = live
    for old, new in ns["LIVE"]:
        new_live = replace_exact(new_live, old, new)
    return replace_exact(src, live, new_live)

def patch_mem(src):
    """release each batched pair-series product buffer before the next one is built (lower transient server peak, same requests)."""
    src = replace_exact(src, "    fnp.sum(X, axis=0, out=Z0)                                            # ans\n",
                        "    fnp.sum(X, axis=0, out=Z0)                                            # ans\n    del X\n")
    src = replace_exact(src, "    T1, T2, T3, T4, T5 = fnp.unstack(fnp.sum(X, axis=1))                  # tri(F2,G1) tri(F1,G2) tri(F2,G2) tri(F3,G1) tri(F1,G3)\n",
                        "    T1, T2, T3, T4, T5 = fnp.unstack(fnp.sum(X, axis=1))                  # tri(F2,G1) tri(F1,G2) tri(F2,G2) tri(F3,G1) tri(F1,G3)\n    del X\n")
    return src

import re
HOLD_FUNCS = ("def _nonlinear_pair_batch(", "def _bd_update(", "def gnn_step(", "def _e2_wpow(", "def _e2_wsums(", "def _message_edge_source(",
              "def _e2_messages(")
_OUT_STMT = re.compile(r"^(\s+)(fnp\.(?:add|subtract|multiply|divide|clip|power|maximum|log1p|tanh|cumprod|sum|stack|outer|copyto)\(.*out=(?!X\)).*\))$")

def patch_hold(src):
    """Every fnp op called with out= returns a fresh alias handle; discarding it makes the client send a hidden `free` round trip
    before the next request. Hold those aliases in a per-call list (released together at function exit = one free trip).
    Aliases share their out buffer, so this costs no server memory; the pair map's transient X product is excluded."""
    for start in HOLD_FUNCS:
        assert src.count(start) >= 1, start
        i = src.rindex(start)                       # the live (last) definition
        j = src.find("\ndef ", i + 1); j = len(src) if j < 0 else j
        k = src.find("\n_E2", i + 1)
        if 0 <= k < j: j = k
        body = src[i:j]
        lines = body.split("\n")
        out, first = [], True
        for ln in lines:
            if first and ln.startswith("def "):
                out.append(ln)
                if ln.rstrip().endswith(":"):
                    out.append("    _e2_hold = []   # E2: out= alias handles, released together at exit (one hidden free trip)")
                    first = False
                continue
            if first and out and not out[-1].rstrip().endswith(":"):
                out.append(ln)
                if ln.rstrip().endswith(":"):
                    out.append("    _e2_hold = []   # E2: out= alias handles, released together at exit (one hidden free trip)")
                    first = False
                continue
            code, comment = ln, ""
            if "   #" in ln:
                code, comment = ln[:ln.index("   #")].rstrip(), ln[ln.index("   #"):]
            parts = [q for q in code.split("; ")]
            if all(_OUT_STMT.match(("    " if n else "") + q if n else q) for n, q in enumerate(parts)) and "out=" in code and code.strip().startswith("fnp."):
                indent = code[:len(code) - len(code.lstrip())]
                stmts = [q.strip() for q in parts]
                out.append(indent + "; ".join("_e2_hold.append(%s)" % q for q in stmts) + comment)
            else:
                out.append(ln)
        new_body = "\n".join(out)
        assert new_body.count("_e2_hold = []") == 1, start
        src = src[:i] + new_body + src[j:]
    src = replace_exact(src, "    F=F/t['feature_scale'];fnp.divide(R,t['rho_scale'],out=R);fnp.divide(S,t['third_scale'],out=S)\n",
                        "    F=F/t['feature_scale'];_e2_hold.append(fnp.divide(R,t['rho_scale'],out=R));_e2_hold.append(fnp.divide(S,t['third_scale'],out=S))\n")
    src = replace_exact(src, "            fnp.outer(vv, vv, out=_e2_VV)\n", "            _e2_VVh = fnp.outer(vv, vv, out=_e2_VV)   # E2: alias held to the end of the layer (no hidden free trip)\n")
    return src

HOLD2_FUNCS = [
    ("def gaussian_raw(mu, v):\n", False), ("def derivatives(a, pdf, g, mg_over_s, order):\n", False),
    ("def features(l, m, C, K21, aux, W):\n", False), ("def silu(x):\n", False), ("def gnn_step(l,x,sigma,W,state,P):\n", False),
    ("def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity'):\n", False),
    ("def nonlinear_scalar(mu, v, k3, k4):\n", False), ("def final_scalar_features(l,m,aux,W):\n", False),
    ("def _nonlinear_first_gaussian(mu, C, K21, k4, k22_override=None, metric='identity'):\n", False),
    ("def omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host):\n", False),
    ("def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):\n", False),
    ("def _e2_bdnh_coefficients(H, A, cstar_all, l):\n", False), ("def _shared_old384_finite(value, name):\n", False),
    ("def _shared_old384_push(layer, w32, p, previous, newborn, lam, state):\n", False),
    ("def _shared_old384_read(left, right, lam, live_indices, state):\n", False),
    ("def _shared_old384_commit(layer, gate, alive, selected, state):\n", False), ("def _slr_inner(old_left, lam_old, state):\n", False),
    ("def _slr_join_factors(layer, overlap, state, Fj):\n", False), ("def _slr_reconstruct(U, state):\n", False),
    ("def _slr_virtual_rotation_envelope(overlap, base):\n", False), ("def _push_bank_shared_newborn(w,p,previous,newborn):\n", False),
    ("def _flms_star(w, A):\n", False), ("def _flms_covariance(w, C):\n", False), ("def _flms_star_diagonal(w, A):\n", False),
    ("def _flms_covariance_diagonal(w, C):\n", False), ("def _message_edge_source(layer,m,K21,aux,w,state,P,context):\n", False),
    ("def _e2_wsums(st):\n", False), ("def _e2_messages(k, W, state):\n", False), ("def _e2_wpow(W, need6=False):\n", False),
    ("def _e2_ms(m, s):\n", False), ("def _cp_diagonal_cube(x):\n", False),
    (LIVE_DEF, True),
]
HOLD2_SKIP = (   # porter's physics anchors stay byte-identical; the live diagonal statement keeps its two n x K temporaries short-lived
    "    fnp.multiply(t4[:, None], rho, out=st['T31'])",
    "            lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])",
    "    common = _w5_port_call(_W5_PORT_SCOPE,w,previous,'newborn_common',_mm,wp,base[previous,:])",
    "            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)",
    "                diagonal = fnp.sum(_e2_pp2 * _e2_pp64 * _e2_lam64, axis=1)",
)

def patch_hold2(src):
    """Hidden-free-trip collapse for expression temporaries: an AST pass wraps every temporary-producing operand in the body
    functions with _e2_H(...), a per-call (per-iteration in the live loop) list released at once. Same operations, same order."""
    sys.path.insert(0, str(HERE)); import hold2
    src = replace_exact(src, "_E2_LAYER = {}", hold2.HOLD_CLASS.strip() + "\n\n\n_E2_LAYER = {}")
    total = 0
    for start, loop in HOLD2_FUNCS:
        src, k = hold2.apply(src, start, HOLD2_SKIP, loop)
        total += k
    print(f"hold2: {total} statements rewritten")
    return src

PATCHES = dict(pair=patch_pair, bd=patch_bd, learned=patch_learned, loop=patch_loop, fuse=patch_fuse, vec=patch_vec, share=patch_share, mem=patch_mem, hold=patch_hold, hold2=patch_hold2)

if __name__ == "__main__":
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(PATCHES)
    src = base.read_text()
    for name in names:
        src = PATCHES[name](src)
    out.write_text(src)
    print(f"built {out.name} from {base.name} with patches {names}")
