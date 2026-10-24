#!/usr/bin/env python3
"""TAG fable-porter (PT-5): PHYSICS PATCH builder. Exact string edits with count asserts that apply to the bank AND to the output of
engineer2's build_e2.py (stacking order: bank -> build_e2.py -> build_physics.py -> engineer kernel fragments). Two features, both inert
unless configured (unconfigured rows/layers run the original statements: bit-exact, no added ops):
 (1) per-row K31 scale in the live pair map (layers 1..14): T31 = (t4 * r)[:, None] * rho, r_a = c0 + c1 * mu_a * k3_a / v_a^2 with
     k3 = diag of the K21 passed in, v = diag of the C passed in; only the T31 statement changes (t4 elsewhere, e.g. the local e4 term, does not);
 (2) extra (3,1) plane: T31 += offdiag(X / (v_a s_a s_c)) with X = _t31_extra(l, ctx) from a formula file (contract: t31_formulas/TEMPLATE.py),
     ctx = the live loop's locals + ctx['_t31_carry']; optional _t31_commit(l, ctx) after each layer's births;
 plus a handle on the newborn push's common product: _PHYS_HANDLES['newborn_common'] = W[previous,:]^T base[previous,:] (f32, n x births of
 the previous layer), refreshed by every push, i.e. current at the pair-map call.
Edits: pair-map def (+ extra31=None); its T31 statement (bank: `T31 = t4[:, None] * rho if k31 else None`; e2:
`fnp.multiply(t4[:, None], rho, out=st['T31'])`, or engineer2's v7 hold form `_e2_hold.append(fnp.multiply(...))`); the live loop's pair-map call (layer slot + masked extra31); the per-network carry dict;
the commit hook after the lam update; the newborn handle in _push_bank_shared_newborn.
usage: build_physics.py <base_cand.py> <out.py> <k31row: none | c0,c1[@layers] | table.json> <x31 formula.py | none> <x31 layers | ''>
       layers: 'a-b' or 'i,j,k' (subset of 1..14; c0,c1 without @ = 1-14). Refuses to overwrite <out.py>."""
import sys, ast, json, hashlib
from pathlib import Path

DEF = "def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity'):\n"
DEF_NEW = "def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity', extra31=None):\n"
T31_BANK = "    T31 = t4[:, None] * rho if k31 else None\n"
T31_BANK_NEW = ("    _r31 = _k31_row_scale(mu, k3, v)\n"
                "    T31 = (t4 * _r31 if _r31 is not None else t4)[:, None] * rho if k31 else None\n"
                "    if extra31 is not None:\n"
                "        x31 = extra31 / (v * s)[:, None] / s[None, :]\n"
                "        fnp.fill_diagonal(x31, 0)\n"
                "        T31 = T31 + x31\n")
T31_E2 = "    fnp.multiply(t4[:, None], rho, out=st['T31'])\n"
T31_E2_NEW = ("    _r31 = _k31_row_scale(mu, k3, v)\n"
              "    if extra31 is None:\n"
              "        fnp.multiply((t4 * _r31 if _r31 is not None else t4)[:, None], rho, out=st['T31'])\n"
              "    else:\n"
              "        x31 = extra31 / (v * s)[:, None] / s[None, :]\n"
              "        fnp.fill_diagonal(x31, 0)\n"
              "        fnp.add((t4 * _r31 if _r31 is not None else t4)[:, None] * rho, x31, out=st['T31'])\n")
T31_E2H = "    _e2_hold.append(fnp.multiply(t4[:, None], rho, out=st['T31']))\n"
T31_E2H_NEW = ("    _r31 = _k31_row_scale(mu, k3, v)\n"
               "    if extra31 is None:\n"
               "        _e2_hold.append(fnp.multiply((t4 * _r31 if _r31 is not None else t4)[:, None], rho, out=st['T31']))\n"
               "    else:\n"
               "        x31 = extra31 / (v * s)[:, None] / s[None, :]\n"
               "        fnp.fill_diagonal(x31, 0)\n"
               "        _e2_hold.append(fnp.add((t4 * _r31 if _r31 is not None else t4)[:, None] * rho, x31, out=st['T31']))\n")
CALL = "            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n"
CALL_NEW = ("            _PHYS_LAYER[0] = l\n"
            "            if l in _T31_LAYERS:\n"
            "                m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric, extra31=_t31_extra(l, locals()))\n"
            "            else:\n"
            "                m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n")
STATE = "        shared_old = _shared_old384_state()\n"
STATE_NEW = STATE + "        _t31_carry = {}\n        _PHYS_HANDLES.clear()\n"
LAM_PREFIX = "            lam = fnp.concatenate([lam, "
COMMIT = "            if _t31_commit is not None:\n                _t31_commit(l, locals())\n"
COMMON = "    common = _w5_port_call(_W5_PORT_SCOPE,w,previous,'newborn_common',_mm,wp,base[previous,:])\n"
COMMON_NEW = COMMON + "    _PHYS_HANDLES['newborn_common'] = common\n"
FORBIDDEN = {'np', 'numpy', 'torch', 'scipy', '__import__', 'open', 'exec', 'eval'}


def span(src, name, last=True):
    starts = [0]
    for line in src.split('\n'):
        starts.append(starts[-1] + len(line) + 1)
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name]
    node = nodes[-1] if last else nodes[0]
    return starts[node.lineno - 1], starts[node.end_lineno], len(nodes)


def edit(src, lo, hi, old, new):
    part = src[lo:hi]
    assert part.count(old) == 1, (old.strip()[:70], part.count(old))
    return src[:lo] + part.replace(old, new) + src[hi:]


def parse_layers(text):
    if not text:
        return ()
    if '-' in text:
        a, b = text.split('-')
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(x) for x in text.split(','))


def parse_k31row(spec):
    if spec == 'none':
        return {}
    if spec.endswith('.json'):
        return {int(k): (float(v[0]), float(v[1])) for k, v in json.loads(Path(spec).read_text()).items()}
    coef, _, layers = spec.partition('@')
    c0, c1 = (float(x) for x in coef.split(','))
    return {l: (c0, c1) for l in parse_layers(layers or '1-14')}


def check_formula(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.Import, ast.ImportFrom)), 'formula must not import'
        assert not (isinstance(node, ast.Name) and node.id in FORBIDDEN), f'formula uses forbidden name {node.id}'
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def patch(src, k31row, formula_text, x31_layers):
    assert '_PHYS_LAYER' not in src, 'base already carries the physics patch'
    assert all(1 <= l <= 14 for l in list(k31row) + list(x31_layers)), 'only layers 1..14 use the live pair map'
    body = ast.parse(src).body
    binds = [n for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'nonlinear' for t in n.targets)]
    assert isinstance(binds[-1].value, ast.Name) and binds[-1].value.id == '_nonlinear_pair_batch', 'live pair map binding'
    lo, hi, count = span(src, '_nonlinear_pair_batch')
    assert count == 1 and src[lo:hi].startswith(DEF)
    pair = src[lo:hi]
    forms = [(T31_BANK, T31_BANK_NEW)] if pair.count(T31_BANK) else [] 
    forms += [(T31_E2, T31_E2_NEW)] if pair.count(T31_E2) else []
    forms += [(T31_E2H, T31_E2H_NEW)] if pair.count(T31_E2H) else []
    assert len(forms) == 1, 'pair map T31 statement: need exactly one known form (bank or engineer2)'
    src = src[:lo] + DEF_NEW + src[lo + len(DEF):]
    lo, hi, _ = span(src, '_nonlinear_pair_batch')
    src = edit(src, lo, hi, *forms[0])
    lo, hi, _ = span(src, '_predict_binary')
    src = edit(src, lo, hi, CALL, CALL_NEW)
    lo, hi, _ = span(src, '_predict_binary')
    src = edit(src, lo, hi, STATE, STATE_NEW)
    lo, hi, _ = span(src, '_predict_binary')
    live = src[lo:hi]
    assert live.count(LAM_PREFIX) == 1, ('lam update', live.count(LAM_PREFIX))
    j = live.index(LAM_PREFIX); j = live.index('\n', j) + 1
    src = src[:lo] + live[:j] + COMMIT + live[j:] + src[hi:]
    lo, hi, _ = span(src, '_push_bank_shared_newborn')
    src = edit(src, lo, hi, COMMON, COMMON_NEW)
    defined = check_formula(formula_text) if formula_text else set()
    if x31_layers:
        assert '_t31_extra' in defined, 'enabled x31 layers need a formula defining _t31_extra'
    src += ("\n\n# fable-porter PT-5 physics patch (build_physics.py): per-row K31 scale + extra (3,1) plane + newborn handle.\n"
            "_PHYS_LAYER = [None]\n_PHYS_HANDLES = {}\n"
            f"_K31R_TABLE = {dict(sorted(k31row.items()))!r}\n"
            "def _k31_row_scale(mu, k3, v):\n"
            "    c = _K31R_TABLE.get(_PHYS_LAYER[0])\n"
            "    if c is None:\n        return None\n"
            "    return c[0] + c[1] * mu * k3 / v ** 2\n"
            f"_T31_LAYERS = {tuple(x31_layers)!r}\n"
            "def _t31_extra(l, ctx):\n    raise RuntimeError('T31 plumbing enabled without a formula')\n"
            "_t31_commit = None\n")
    if formula_text:
        src += "\n# ---- formula (appended verbatim) ----\n" + formula_text.rstrip('\n') + "\n"
    ast.parse(src)
    return src, {T31_BANK: 'bank', T31_E2: 'e2', T31_E2H: 'e2-hold'}[forms[0][0]]


if __name__ == '__main__':
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    k31row, formula, x31_layers = parse_k31row(sys.argv[3]), sys.argv[4], parse_layers(sys.argv[5])
    ftext = None if formula == 'none' else Path(formula).read_text()
    text, form = patch(base.read_text(), k31row, ftext, x31_layers)
    with out.open('x') as f:
        f.write(text)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    print(json.dumps(dict(out=str(out), base=str(base), base_sha256=sha(base.read_bytes()), pair_form=form, k31row={str(k): v for k, v in sorted(k31row.items())},
                          formula=formula, x31_layers=list(x31_layers), out_sha256=sha(text.encode()))))
