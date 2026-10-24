#!/usr/bin/env python3
"""TAG fable-engineer2 (E-71 group 5 BODY DIET builder --g5, ROUTE B ONLY, self-contained v2, 2026-09-25 (post-mortem: no gnn_step chain; <= 3 einsum operands; no scalar-value operands); lead GO for S26, before treuse): <out> = <base> + g5_head.py helpers + the body
functions below re-emitted from <base>'s LIVE source with asserted one-to-one replacements (value reuse, einsum merges, per-process
constants), + alias rebinds (nonlinear, _gn_message_edge_source_base) + a setup hook binding the constants. Optional replacements (lines a
later layer such as j3cut may have removed) are reported, never silently skipped. usage: build_g5_B_v2.py <base.py> <out.py>
Route A re-emits the pair map and the live loop (full value reuse; needs the live-definition einsum gate, einsum_site_diff_live.py).
Route B re-emits neither (porter's einsum gate as is): _gn_dcov reads the pair state's persistent float32 buffers instead."""
import ast, sys, re, hashlib
G5_HEAD = "\n\n# BEGIN FABLE-ENGINEER2 E-71 GROUP 5 BODY DIET (--g5, 2026-09-25; lead GO for S26 as the k-split package's request offset). BIT-IDENTICAL\n# arithmetic (verified by setup-first parity on 4 nets, all 16 layers). Three levers, all in the physics body (no kernel function is touched):\n#  (1) VALUE REUSE: the pair map publishes values it already computes (Hermite rows aa, the K31 row scale, k4/v^2, rho, h4*pdf, k3/s^3, and\n#      the live loop's diag(Cpre)); _gn_dcov and the message edge read them under object-identity checks (else the original code path), and\n#      the live loop computes (3u)u once. Every reused value is the same operation on the same operands as the recomputation it replaces.\n#  (2) EINSUM MERGES: a left-deep multiply chain and the broadcast views feeding it run as ONE fnp.einsum(..., optimize=False): numpy\n#      multiplies its operands left to right, so the products are bit-identical (IEEE multiply commutes exactly); an exact -0.0 product\n#      can come out +0.0. Only DISTINCT operand objects (crash law, checked at run time), no out= (flopscope bills it 2x), and only\n#      forms measured no slower on the server (vector chains; one multiply plus its views; float64 patterns measured faster).\n#  (3) CONSTANTS: scalars and small literal arrays are per-process 0-d / 1-d constants bound in setup (off-budget).\n# Built by fable_engineer2_v1/diet5/g5/build_g5.py from the base's live source (asserted replacements).\n_G5_CONST = {}\n\n\ndef _g5c(value, dtype):\n    key = ('s', float(value), str(dtype))\n    c = _G5_CONST.get(key)\n    if c is None:\n        c = _G5_CONST[key] = fnp.asarray(float(value), dtype=str(dtype))\n    return c\n\n\ndef _g5_arr(values):\n    # a per-process float64 literal vector (the same bytes fnp.asarray(list, dtype=float64) makes per call)\n    key = ('a', tuple(float(v) for v in values))\n    c = _G5_CONST.get(key)\n    if c is None:\n        c = _G5_CONST[key] = fnp.asarray(list(key[1]), dtype=fnp.float64)\n    return c\n\n\ndef _g5_j1(values, phys):\n    # per-process (_j1m, phys * (1.0 - _j1m), that as a column view): the constants of the birth update's j1 physical leg\n    key = ('j1', tuple(float(v) for v in values), float(phys))\n    c = _G5_CONST.get(key)\n    if c is None:\n        m = _g5_arr(values)\n        tail = phys * (1.0 - m)\n        c = _G5_CONST[key] = (m, tail, tail[:, None])\n    return c\n\n\ndef _g5_sub(x, n):\n    k = len(x.shape)\n    if k == 0:\n        return ''\n    if k == 1 and int(x.shape[0]) in (n, 1):\n        return 'i'\n    return None\n\n\ndef _g5_chain(*ops):\n    # ops[0] * ops[1] * ops[2] left to right (<= 3 operands: flopscope's einsum is left to right only up to 3); 0-d, (n,) or (1,) ARRAYS (no scalar values), one dtype (a bool mask allowed LAST), distinct objects\n    arrays = [x for x in ops]\n    shaped = len(arrays) <= 3 and all(hasattr(x, 'shape') and hasattr(x, 'dtype') and type(x).__name__ != 'RemoteScalar' for x in arrays)   # v2: flopscope runs 4+ operands pairwise; scalar values are not arrays\n    vec = [int(x.shape[0]) for x in arrays if shaped and len(x.shape) == 1]\n    n = max(vec) if vec else 0\n    subs = [_g5_sub(x, n) for x in arrays] if shaped and vec else [None]\n    dts = [str(x.dtype) for x in arrays] if shaped else ['?']\n    ok = (None not in subs and len({id(x) for x in arrays}) == len(arrays) and len(set(dts[:-1])) == 1\n          and dts[-1] in (dts[0], 'bool') and dts[0] in ('float32', 'float64'))\n    if ok:\n        return fnp.einsum(','.join(subs) + '->i', *arrays, optimize=False)\n    out = arrays[0]\n    for x in arrays[1:]:\n        out = out * x\n    return out\n\n\ndef _g5_schain(scale, *ops):\n    # (scale * ops[0]) * ops[1] * ...; scale is a Python number (a per-process 0-d constant) or a 0-d array\n    if isinstance(scale, (int, float)):\n        dt = str(getattr(ops[0], 'dtype', ''))\n        if scale != 0 and dt in ('float32', 'float64'):   # a zero scale keeps the original multiplies (no constant; signed zeros as before)\n            return _g5_chain(_g5c(scale, dt), *ops)\n        out = scale * ops[0]\n        for x in ops[1:]:\n            out = out * x\n        return out\n    return _g5_chain(scale, *ops)\n\n\ndef _g5_rowmul(x, M):\n    # x[:, None] * M\n    if len(x.shape) == 1 and len(M.shape) == 2 and int(x.shape[0]) == int(M.shape[0]) and x is not M:\n        return fnp.einsum('i,ij->ij', x, M, optimize=False)\n    return x[:, None] * M\n\n\ndef _g5_mulrow(M, x):\n    # M * x[:, None]\n    if len(x.shape) == 1 and len(M.shape) == 2 and int(x.shape[0]) == int(M.shape[0]) and x is not M:\n        return fnp.einsum('ij,i->ij', M, x, optimize=False)\n    return M * x[:, None]\n\n\ndef _g5_outer(x, y):\n    # x[:, None] * y[None, :]\n    if len(x.shape) == 1 and len(y.shape) == 1 and x is not y:\n        return fnp.einsum('i,j->ij', x, y, optimize=False)\n    return x[:, None] * y[None, :]\n\n\ndef _g5_bcast3(M, P):\n    # M * P[:, None, :]   (M (n, n), P (a, n))\n    if len(M.shape) == 2 and len(P.shape) == 2 and int(P.shape[1]) == int(M.shape[1]) and M is not P:\n        return fnp.einsum('ij,aj->aij', M, P, optimize=False)\n    return M * P[:, None, :]\n\n\ndef _g5_srowmul(scale, x, M):\n    # einsum('ij,i->ij', M, scale * x) == (scale * x)[:, None] * M  ->  one einsum over (scale, x, M) (float64 form, measured faster)\n    if (len(x.shape) == 1 and len(M.shape) == 2 and int(x.shape[0]) == int(M.shape[0]) and x is not M\n            and str(x.dtype) == str(M.dtype) == 'float64'):\n        return fnp.einsum(',i,ij->ij', _g5c(scale, 'float64'), x, M, optimize=False)\n    return fnp.einsum('ij,i->ij', M, scale * x, optimize=False)\n\n\ndef _g5_stack_ss(S, s, pdf):\n    # (S * s) * pdf with S (a, n) and s, pdf (n,) (float64 form, measured faster)\n    if (len(S.shape) == 2 and len(s.shape) == 1 and len(pdf.shape) == 1 and s is not pdf\n            and str(S.dtype) == str(s.dtype) == str(pdf.dtype) == 'float64'):\n        return fnp.einsum('aj,j,j->aj', S, s, pdf, optimize=False)\n    return (S * s) * pdf\n"
from pathlib import Path
HERE = Path(__file__).resolve().parent
base_path, out_path = sys.argv[1], sys.argv[2]
ROUTE = 'B'   # the lead's route (no re-emit of the pair map or the live loop; the porter's einsum gate unchanged)
src = open(base_path).read(); lines = src.split('\n')
tree = ast.parse(src)
defs = {}
for n in tree.body:
    if isinstance(n, ast.FunctionDef):
        defs.setdefault(n.name, []).append(n)
def text(n): return '\n'.join(lines[n.lineno - 1 - len(n.decorator_list):n.end_lineno])
def live(name): return text(defs[name][-1])
report = []
def rep(t, old, new, name, count=1, optional=False):
    c = t.count(old)
    if c != count:
        if optional and c == 0:
            report.append(f"  {name}: OPTIONAL replacement absent: {old.strip()[:90]}")
            return t
        raise SystemExit(f"{name}: expected {count} of {old.strip()[:100]!r}, found {c}")
    report.append(f"  {name}: applied x{count}: {old.strip()[:90]}")
    return t.replace(old, new)
out = []
# 1. pair map (+ rebind the import-time alias nonlinear): route A only
if ROUTE == 'A':
    f = live('_nonlinear_pair_batch')
    f = rep(f, "    v = fnp.diag(C)\n", "    v = _E2_LAYER['vv'] if (_E2_LAYER.get('Cpre') is C and _E2_LAYER.get('vv') is not None) else fnp.diag(C)   # G5: the live loop's diag(Cpre)\n", 'pair')
    f = rep(f, "_e2_H(_e2_H(_e2_H(fnp.stack((st['one'], mu, q2, mq5))) * s) * pdf)", "_e2_H(_g5_stack_ss(_e2_H(fnp.stack((st['one'], mu, q2, mq5))), s, pdf))", 'pair')
    f = rep(f, "post_k3 = _e2_H(r3 - _e2_H(_e2_H(3 * m) * r2))", "post_k3 = _e2_H(r3 - _e2_H(_g5_schain(3, m, r2)))", 'pair')
    f = rep(f, "_e2_H(r4 - _e2_H(_e2_H(4 * m) * r3))", "_e2_H(r4 - _e2_H(_g5_schain(4, m, r3)))", 'pair')
    f = rep(f, "    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d, ss=ss, mgs=mgs)\n",
               "    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d, ss=ss, mgs=mgs, aa=aa, r31=_r31, t4=t4, rho=rho, A4=A4, mu=mu, K21=K21, k4=k4)   # G5: reused under identity checks\n", 'pair')
    out.append(f + "\n\n\nnonlinear = _nonlinear_pair_batch   # G5: the live loop reaches the pair map through this import-time alias")
# 2. _gn_dcov: fast path on the pair map's published values (A) or its persistent pair-state buffers (B), else the original
if ROUTE == 'A':
    f = live('_gn_dcov')
    f = rep(f, "    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)\n",
               "    pr = _E2_LAYER.get('pair')\n"
               "    if (pr is None or 'aa' not in pr or pr['aux'] is not aux or pr['mu'] is not mu or pr['K21'] is not K21 or pr['k4'] is not k4\n"
               "            or aux['pre_cov'] is not Cpre or _S19_GN_TORCH or '_k31_row_scale' not in globals()):\n"
               "        return _G5_OLD_GN_DCOV(l, mu, Cpre, K21, k4, aux)\n"
               "    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)\n", 'gn')
    f = rep(f, "    v = _e2_H(fnp.diag(Cpre)).copy()\n    A, _unused = derivatives(a, aux['pdf'], aux['gate'], _e2_H(aux['gaussian_mean'] / s), 5)\n    A1, A2, A3, A4, A5 = [x.astype(fnp.float32) for x in A[1:6]]\n    ss = fnp.outer(s, s)\n    rho = _e2_H(Cpre / ss).astype(fnp.float32)\n    fnp.fill_diagonal(rho, 0)\n",
               "    v = aux['pre_v']   # G5: = diag(Cpre), the pair map's\n"
               "    A1, A2, A3, A4, A5 = fnp.unstack(_e2_H(_e2_H(pr['aa'][1:6]).astype(fnp.float32)))   # G5: = derivatives(a, pdf, g, mg/s, 5)[1:6] (the pair map's rows)\n"
               "    ss = pr['ss']   # G5: = outer(s, s)\n"
               "    rho = pr['rho'].astype(fnp.float32)   # G5: (Cpre / ss) with a zero diagonal, cast\n", 'gn')
    f = rep(f, "    k3 = _e2_H(fnp.diag(K21)).copy()\n", "    k3 = aux['pre_k3']   # G5: = diag(K21)\n", 'gn')
    f = rep(f, "    dK21 = _e2_H(_e2_H(_e2_H(_e2_H(_e2_H(_e2_H(th['iso']) * kink) * k3) / v)[:, None]) * Cpre) + _e2_H(_e2_H(_e2_H(_e2_H(th['k21n']) * _e2_H(1 - kink))[:, None]) * K21)",
               "    dK21 = _e2_H(_g5_rowmul(_e2_H(_e2_H(_g5_schain(th['iso'], kink, k3)) / v), Cpre)) + _e2_H(_g5_rowmul(_e2_H(_e2_H(th['k21n']) * _e2_H(1 - kink)), K21))", 'gn')
    f = rep(f, "    dT21 = _e2_H(dK21 / _e2_H(_e2_H(v[:, None]) * _e2_H(s[None, :]))).astype(fnp.float32)",
               "    dT21 = _e2_H(dK21 / _e2_H(_g5_outer(v, s))).astype(fnp.float32)", 'gn')
    f = rep(f, "    t4 = _e2_H(fnp.broadcast_to(k4, (len(mu),))) / _e2_H(v ** 2)\n    r31 = ((_k31_row_scale_gn(mu, k3, v) if _S19_GN_TORCH else _k31_row_scale(mu, k3, v))\n           if '_k31_row_scale' in _e2_H(globals()) else None)   # PT-33 S19: torch GN remainder keeps the configure-time 0.85/0.5\n",
               "    t4 = pr['t4']   # G5: = k4 / v**2\n    r31 = pr['r31']   # G5: = _k31_row_scale(mu, k3, v)\n", 'gn')
    f = rep(f, "    dT31 = _e2_H(_e2_H(_e2_H(_e2_H(th['k31']) * (t4 * r31 if r31 is not None else t4)).astype(fnp.float32))[:, None]) * rho",
               "    dT31 = _g5_rowmul(_e2_H(_e2_H(_g5_chain(t4, r31, _g5c(th['k31'], 'float64')) if r31 is not None else _e2_H(th['k31']) * t4).astype(fnp.float32)), rho)   # G5: k31*(t4*r31) = (t4*r31)*k31", 'gn')
else:
    f = live('_gn_dcov')
    f = rep(f, "    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)\n",
               "    pr = _E2_LAYER.get('pair')\n"
               "    if pr is None or pr.get('aux') is not aux or aux['pre_cov'] is not Cpre or aux.get('pre_k3') is None:\n"
               "        return _G5_OLD_GN_DCOV(l, mu, Cpre, K21, k4, aux)\n"
               "    _g5st = _e2_pair_state(int(mu.shape[0]))\n"
               "    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)\n", 'gnB')
    f = rep(f, "    v = _e2_H(fnp.diag(Cpre)).copy()\n    A, _unused = derivatives(a, aux['pdf'], aux['gate'], _e2_H(aux['gaussian_mean'] / s), 5)\n    A1, A2, A3, A4, A5 = [x.astype(fnp.float32) for x in A[1:6]]\n    ss = fnp.outer(s, s)\n    rho = _e2_H(Cpre / ss).astype(fnp.float32)\n    fnp.fill_diagonal(rho, 0)\n",
               "    v = aux['pre_v']   # G5: = diag(Cpre), the pair map's\n"
               "    A1, A2, A3, A4, A5 = fnp.unstack(_e2_H(_g5st['FT'][1:6, 0]))   # G5: the pair map stacked aa.astype(float32) = derivatives(a, pdf, g, mg/s, 6) rows into FT[:, 0]\n"
               "    ss = pr['ss']   # G5: = outer(s, s)\n"
               "    rho = _g5st['P16'][0]   # G5: = (Cpre / ss) with a zero diagonal, cast: the pair map's rho.astype(float32) / 1\n", 'gnB')
    f = rep(f, "    k3 = _e2_H(fnp.diag(K21)).copy()\n", "    k3 = aux['pre_k3']   # G5: = diag(K21)\n", 'gnB')
    f = rep(f, "    dK21 = _e2_H(_e2_H(_e2_H(_e2_H(_e2_H(_e2_H(th['iso']) * kink) * k3) / v)[:, None]) * Cpre) + _e2_H(_e2_H(_e2_H(_e2_H(th['k21n']) * _e2_H(1 - kink))[:, None]) * K21)",
               "    dK21 = _e2_H(_g5_rowmul(_e2_H(_e2_H(_g5_schain(th['iso'], kink, k3)) / v), Cpre)) + _e2_H(_g5_rowmul(_e2_H(_e2_H(th['k21n']) * _e2_H(1 - kink)), K21))", 'gnB')
    f = rep(f, "    dT21 = _e2_H(dK21 / _e2_H(_e2_H(v[:, None]) * _e2_H(s[None, :]))).astype(fnp.float32)",
               "    dT21 = _e2_H(dK21 / _e2_H(_g5_outer(v, s))).astype(fnp.float32)", 'gnB')
    f = rep(f, "    dT31 = _e2_H(_e2_H(_e2_H(_e2_H(th['k31']) * (t4 * r31 if r31 is not None else t4)).astype(fnp.float32))[:, None]) * rho",
               "    dT31 = _g5_rowmul(_e2_H(_e2_H(_g5_chain(t4, r31, _g5c(th['k31'], 'float64')) if r31 is not None else _e2_H(th['k31']) * t4).astype(fnp.float32)), rho)   # G5: k31*(t4*r31) = (t4*r31)*k31", 'gnB')
out.append("_G5_OLD_GN_DCOV = _gn_dcov\n\n\n" + f)
# 3. _bd_update
f = live('_bd_update')
f = rep(f, "    gate_jet = _e2_H(_e2_H(skew * _e2_H(_e2_H(alpha * alpha) - 1)) * aux['pdf']) / 6", "    gate_jet = _e2_H(_g5_chain(skew, _e2_H(_e2_H(alpha * alpha) - 1), aux['pdf'])) / 6", 'bd', optional=True)
m = re.search(r"        _j1m = fnp\.asarray\(\[([0-9., ]+)\], dtype=fnp\.float64\)[^\n]*\n", f)
assert m, 'bd: _j1m literal not found'
vals = tuple(float(x) for x in m.group(1).split(','))
f = f.replace(m.group(0), f"        _j1m, _g5_j1tail, _g5_j1col = _g5_j1({vals!r}, _BD_J1_PHYS)   # G5: per-process constants\n"); report.append(f"  bd: applied _j1m constants {vals}")
f = rep(f, "        projection = projection * _j1m[:, None]", "        projection = _g5_mulrow(projection, _j1m)", 'bd')
f = rep(f, "        c = (c * _j1m + _BD_J1_PHYS * (1.0 - _j1m)) if c.ndim == 1 else (c * _j1m[:, None] + (_BD_J1_PHYS * (1.0 - _j1m))[:, None])",
           "        c = (c * _j1m + _g5_j1tail) if c.ndim == 1 else (_g5_mulrow(c, _j1m) + _g5_j1col)", 'bd')
f = rep(f, "    _e2_hold.append(fnp.subtract(D, bp * projection[:, None, :], out=D))", "    _e2_hold.append(fnp.subtract(D, _g5_bcast3(bp, projection), out=D))", 'bd')
f = rep(f, "    weight = _e2_H(a64 * _e2_H(t64 - s64)) * births", "    weight = _g5_chain(a64, _e2_H(t64 - s64), births)", 'bd')
f = rep(f, "    quadratic_weight = _e2_H(a64 * _e2_H(_e2_H(t64 * t64) - _e2_H(s64 * s64))) * births", "    quadratic_weight = _g5_chain(a64, _e2_H(_e2_H(t64 * t64) - _e2_H(s64 * s64)), births)", 'bd')
out.append(f)
# 4. derivatives, 5. _k31_row_scale
f = rep(live('derivatives'), "        A.append(_e2_H(_e2_H(_e2_H(_e2_H(-1) ** _e2_H(r - 2)) * h) * pdf))", "        A.append(_e2_H(_g5_schain(_e2_H(-1) ** _e2_H(r - 2), h, pdf)))", 'derivatives')
out.append(f)
f = rep(live('_k31_row_scale'), "    return c[0] + _e2_H(_e2_H(_e2_H(c[1] * mu) * k3) / _e2_H(v ** 2))", "    return c[0] + _e2_H(_e2_H(_g5_schain(c[1], mu, k3)) / _e2_H(v ** 2))", 'k31')
out.append(f)
# 6. the live loop: route A only
if ROUTE == 'A':
    f = live('_predict_binary')
    f = rep(f, "            _E2_LAYER.update(mu=mu, Cpre=Cpre, vc=_e2_vc, s=_e2_s, a=_e2_a, g=response_gate)\n", "            _E2_LAYER.update(mu=mu, Cpre=Cpre, vc=_e2_vc, s=_e2_s, a=_e2_a, g=response_gate, vv=vv)   # G5: vv = diag(Cpre) for the pair map\n", 'loop')
    f = rep(f, "_e2_H(fnp.einsum('ij,i->ij', Bpre, _e2_H(2 * bpre), optimize=False))", "_e2_H(_g5_srowmul(2, bpre, Bpre))", 'loop')
    f = rep(f, "_e2_H(fnp.einsum('ij,i->ij', Bpost, _e2_H(2 * b), optimize=False))", "_e2_H(_g5_srowmul(2, b, Bpost))", 'loop')
    f = rep(f, "_e2_H(_e2_H(_e2_H(radial * d) * vv) / 6)", "_e2_H(_e2_H(_g5_schain(radial, d, vv)) / 6)", 'loop', count=2)
    f = rep(f, "            disc = _e2_H(_e2_H(4 * innovation) / safe_birth) - _e2_H(_e2_H(3 * u) * u)\n",
               "            _g5_3uu = _e2_H(_e2_H(3 * u) * u)   # G5: once (born's diagonal reuses it)\n            disc = _e2_H(_e2_H(4 * innovation) / safe_birth) - _g5_3uu\n", 'loop')
    f = rep(f, "_e2_H(_e2_bb * _e2_H(_e2_H(_e2_H(3 * u) * u) + _e2_H(delta * delta)))", "_e2_H(_e2_bb * _e2_H(_g5_3uu + _e2_H(delta * delta)))", 'loop')
    f = rep(f, "                k3 = k3 + _e2_H(_e2_H(3 * active_v) * bpre)", "                k3 = k3 + _e2_H(_g5_schain(3, active_v, bpre))", 'loop')
    out.append(f)
# 7. the message edge BASE (the inner _message_edge_source that the GN wrapper reaches through _gn_message_edge_source_base)
cands = [n for n in defs['_message_edge_source'] if 'F = fnp.stack((g, p,' in text(n)]
assert len(cands) == 1, 'edge base not unique'
f = text(cands[0]).replace('def _message_edge_source(', 'def _g5_message_edge_base(', 1)
if ROUTE == 'A':
    f = rep(f, "_e2_H(_e2_H(_e2_H(a * a) - 1) * p)", "(_g5pr['A4'] if _g5ok else _e2_H(_e2_H(_e2_H(a * a) - 1) * p))", 'edge')
f = rep(f, "_e2_H(_e2_H(sk * a) * p)", "_e2_H(_g5_chain(sk, a, p))", 'edge')
f = rep(f, "_e2_H(_e2_H(ku * a) * p)", "_e2_H(_g5_chain(ku, a, p))", 'edge')
if ROUTE == 'A':
    f = rep(f, "    sk=aux['pre_k3']/s**3;ku=aux['pre_k4']/s**4;ms=_e2_ms(m,s)\n",
               "    sk=aux['pre_k3']/s**3;ku=aux['pre_k4']/s**4;ms=_e2_ms(m,s)\n    _g5pr = _E2_LAYER.get('pair'); _g5ok = _g5pr is not None and _g5pr.get('aux') is aux and 'A4' in _g5pr   # G5: h4*pdf from the pair map\n", 'edge')
out.append(f + "\n\n\n_gn_message_edge_source_base = _g5_message_edge_base   # G5: the GN wrapper calls the base through this alias")
# 8. gnn_step, 9. omitted_radial_source
# (v2) gnn_step is NOT edited: its chain has 4 operands (flopscope executes 4+ einsum operands pairwise) and c['rms'] is a scalar value in client mode.
f = live('omitted_radial_source')
f = rep(f, "    contraction = fnp.sum(_e2_H(_e2_H(_e2_H(lam64.astype(fnp.float64)) * row_energy) * column_sum))", "    contraction = fnp.sum(_e2_H(_g5_chain(_e2_H(lam64.astype(fnp.float64)), row_energy, column_sum)))", 'omitted')
f = rep(f, "    contraction = contraction - _e2_H(fnp.sum(_e2_H(_e2_H(diagonal * a) * beta)))", "    contraction = contraction - _e2_H(fnp.sum(_e2_H(_g5_chain(diagonal, a, beta))))", 'omitted')
out.append(f)
TAIL = f'''

_G5_OLD_SETUP = Estimator.setup
_G5_OLD_RESET = _pb_reset


def _g5_setup(self, context):
    _G5_OLD_SETUP(self, context)
    if int(getattr(context, 'width', 0)) != 1024 or int(getattr(context, 'depth', 0)) != 16:
        return
    def bind():
        for v in sorted({{1.0, -1.0, 2.0, 3.0, 4.0, float(_GN_KNOBS['iso']), float(_GN_KNOBS['k31'])}} | {{float(c[1]) for c in _K31R_TABLE.values()}}):
            _g5c(v, 'float64')
        _g5_j1({vals!r}, _BD_J1_PHYS)
    try:
        try:
            bind()
        except Exception as exc:
            if type(exc).__name__ != 'NoBudgetContextError':
                raise
            with flops.BudgetContext(flop_budget=2 ** 40, quiet=True, namespace='g5const'):
                bind()
    except Exception:
        _G5_CONST.clear()   # never fail setup: the constants then bind lazily in predict


def _pb_reset(saved):
    _G5_CONST.clear()
    return _G5_OLD_RESET(saved)


Estimator.setup = _g5_setup
# END FABLE-ENGINEER2 E-71 GROUP 5 BODY DIET
'''
frag = G5_HEAD + '\n\n' + '\n\n\n'.join(out) + '\n' + TAIL
full = src.rstrip('\n') + '\n' + frag
ast.parse(full)
open(out_path, 'w').write(full)
fp = out_path.replace('.py', '_fragment.py'); open(fp, 'w').write(frag)
print('\n'.join(report))
print(out_path, hashlib.sha256(full.encode()).hexdigest()[:8], '| fragment', fp, hashlib.sha256(frag.encode()).hexdigest()[:8], f'| {len(frag.splitlines())} lines')
