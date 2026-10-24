#!/usr/bin/env python3
"""TAG fable-porter (PT-4): GENERIC native plumbing for a plane-valued correction X to the live pair map's (3,1) input.
At the layers in <layers> (a subset of 1..14, the layers whose pair map is `nonlinear` = _nonlinear_pair_batch) the live loop calls a
patched copy _nonlinear_pair_batch_t31(mu, Cpre, K21, k4, k22, metric, X) whose (3,1) plane is
    T31 = t4[:, None] * rho  +  offdiag( X / ((v * s)[:, None] * s[None, :]) )          v, s = pre-activation variance, sd
so X[a, c] is an additive correction to K31[a, c] = kappa4(h_a, h_a, h_a, h_c) in raw units (a = the cubed unit; the bank's own model is
K31[a, c] = k4_a C_ac / v_a = radial * vv_a * Cpre_ac). Nothing else changes: K21 and every other consumer are untouched.
X = _t31_extra(l, ctx), ctx = the live loop's locals at the pair-map call (names assigned later in a layer still hold the previous
layer's values; ctx['_t31_carry'] is a per-network dict for carried state). Optional _t31_commit(l, ctx) runs right after the layer's
births are committed (lam extended), for carried per-column state. Both come from <formula.py>: flopscope-only source appended to the
candidate (no imports, no numpy/torch); it may define its own literals and helpers.
Disabled (empty layer set, formula none) = the bank's exact op sequence (bit-exact). Plumbing cost per enabled layer: 3 n^2 + n FLOPs
plus the formula's own ops (measure with t31_bill.py).
usage: build_t31_variant.py <base_cand.py> <formula.py | none> <layers: '' | 'a-b' | 'i,j,k'> <out.py>   (refuses to overwrite)"""
import sys, ast, json, hashlib
from pathlib import Path

CALL = "            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n"
CALL_NEW = ("            if l in _T31_LAYERS:\n"
            "                m, C, target, radial_new, aux = _nonlinear_pair_batch_t31(mu, Cpre, K21, k4, k22, metric, _t31_extra(l, locals()))\n"
            "            else:\n"
            "                m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n")
STATE = "        shared_old = _shared_old384_state()\n"
STATE_NEW = STATE + "        _t31_carry = {}\n"
LAM = "            lam = fnp.concatenate([lam, a[selected].astype(fnp.float32), -a[selected].astype(fnp.float32)])\n"
LAM_NEW = LAM + "            if _t31_commit is not None:\n                _t31_commit(l, locals())\n"
DEF = "def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity'):\n"
DEF_NEW = "def _nonlinear_pair_batch_t31(mu, C, K21, k4, k22_override, metric, extra31):\n"
T31 = "    T31 = t4[:, None] * rho if k31 else None\n"
T31_NEW = (T31 +
           "    x31 = extra31 / (v * s)[:, None] / s[None, :]\n"
           "    fnp.fill_diagonal(x31, 0)\n"
           "    T31 = T31 + x31\n")
FORBIDDEN = {'np', 'numpy', 'torch', 'scipy', '__import__', 'open', 'exec', 'eval'}


def spans(src):
    starts = [0]
    for line in src.split('\n'):
        starts.append(starts[-1] + len(line) + 1)
    body = ast.parse(src).body
    live = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == '_predict_binary'][-1]
    pair = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == '_nonlinear_pair_batch']
    binds = [n for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'nonlinear' for t in n.targets)]
    assert len(pair) == 1 and isinstance(binds[-1].value, ast.Name) and binds[-1].value.id == '_nonlinear_pair_batch', 'live pair map binding'
    assert binds[-1].lineno > pair[0].lineno
    return ((starts[live.lineno - 1], starts[live.end_lineno]), (starts[pair[0].lineno - 1], starts[pair[0].end_lineno]))


def check_formula(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.Import, ast.ImportFrom)), 'formula must not import'
        assert not (isinstance(node, ast.Name) and node.id in FORBIDDEN), f'formula uses forbidden name {node.id}'
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)} | \
           {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}


def patch(src, formula_text, layers):
    assert all(1 <= l <= 14 for l in layers), 'only layers 1..14 use the live pair map'
    assert '_T31_LAYERS' not in src, 'base already carries the T31 plumbing'
    (lo, hi), (plo, phi) = spans(src)
    live, pair = src[lo:hi], src[plo:phi]
    for old in (CALL, STATE, LAM):
        assert live.count(old) == 1, old.strip()
    live = live.replace(CALL, CALL_NEW).replace(STATE, STATE_NEW).replace(LAM, LAM_NEW)
    assert pair.startswith(DEF) and pair.count(T31) == 1
    copy = DEF_NEW + pair[len(DEF):].replace(T31, T31_NEW)
    defined = check_formula(formula_text) if formula_text else set()
    if layers:
        assert '_t31_extra' in defined, 'enabled layers need a formula defining _t31_extra'
    out = (src[:lo] + live + src[hi:] +
           "\n\n# fable-porter PT-4: (3,1)-plane correction plumbing (build_t31_variant.py). Live pair map copy with T31 += offdiag(X/(v_a s_a s_c)).\n"
           + copy +
           f"\n\n_T31_LAYERS = {tuple(layers)!r}\n"
           "def _t31_extra(l, ctx):\n    raise RuntimeError('T31 plumbing enabled without a formula')\n"
           "_t31_commit = None\n")
    if formula_text:
        out += "\n# ---- formula (appended verbatim) ----\n" + formula_text.rstrip('\n') + "\n"
    ast.parse(out)
    return out


def parse_layers(text):
    if not text:
        return ()
    if '-' in text:
        a, b = text.split('-')
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(x) for x in text.split(','))


if __name__ == '__main__':
    base, formula, layers, out = Path(sys.argv[1]), sys.argv[2], parse_layers(sys.argv[3]), Path(sys.argv[4])
    ftext = None if formula == 'none' else Path(formula).read_text()
    text = patch(base.read_text(), ftext, layers)
    with out.open('x') as f:
        f.write(text)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    print(json.dumps(dict(out=str(out), base=str(base), base_sha256=sha(base.read_bytes()), formula=formula,
                          formula_sha256=sha(ftext.encode()) if ftext else None, layers=list(layers), out_sha256=sha(text.encode()))))
