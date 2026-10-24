#!/usr/bin/env python3
"""TAG fable-engineer2 (j3cut, 2026-09-25; lead: analyst's ablation survivor, RESULT.md sec. 36): builds <out> = <base> + a fragment that DROPS
the birth update's gate-jet leg j = 3 (direction Cpre * gate_jet, its learned c3 * orth(dir3) term and its physical X3 raw add) with the full
native saving: the persistent direction stack shrinks from (4,n,n) to (3,n,n) (_e2_bd_state3), the gate-jet chain (sigma^3, skew, gate_jet)
is not computed, the coefficients use rows 0..2 (the node head still computes all four columns, so rows 0..2 are unchanged), and the S19 raw
adds use j = 0..2. Directions 0..2 are orthogonalized, weighted and summed exactly as before (per-direction elementwise ops; the ordered sum
over directions is the same prefix), so the ONLY change is the j3 term. An APPROXIMATION (analyst, torch VALID16: raw +0.002%); bill -0.091%
(engineer2 census). _bd_update is rebuilt from <base>'s live source with asserted one-to-one replacements. usage: build_j3cut.py <base.py> <out.py>"""
import ast, sys, hashlib
src_path, out_path = sys.argv[1], sys.argv[2]
src = open(src_path).read()
last = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
lines = src.split('\n')
n = last['_bd_update']
b = '\n'.join(lines[n.lineno - 1 - len(n.decorator_list):n.end_lineno])
def rep(text, old, new):
    assert text.count(old) == 1, (old[:90], text.count(old))
    return text.replace(old, new)
b = rep(b, "    st = _e2_bd_state(n)\n    D = st['D']; D0, D1, D2, D3 = st['Dv']\n    c = coefficients.astype(fnp.float64)",
           "    st = _e2_bd_state3(n)   # J3CUT: three directions\n    D = st['D']; D0, D1, D2 = st['Dv']\n    c = coefficients.astype(fnp.float64)[:3]   # J3CUT: rows 0..2 of the unchanged four-column head")
b = rep(b, "    sigma3 = _e2_H(sigma * sigma) * sigma\n    skew = aux['pre_k3'] / sigma3\n    gate_jet = _e2_H(_e2_H(skew * _e2_H(_e2_H(alpha * alpha) - 1)) * aux['pdf']) / 6\n",
           "")
b = rep(b, "    _e2_hold.append(fnp.multiply(Cpre, gate_jet[:, None], out=D3))\n", "")
b = rep(b, "        for _s19_j, _s19_x in enumerate(_S19_BD_ADD):",
           "        for _s19_j, _s19_x in enumerate(_S19_BD_ADD[:3]):   # J3CUT: no X3 raw leg")
b = rep(b, "        _j1m = fnp.asarray([1.0, 0.0, 1.0, 1.0], dtype=fnp.float64)",
           "        _j1m = fnp.asarray([1.0, 0.0, 1.0], dtype=fnp.float64)   # J3CUT: three legs")
FRAG = '''

# BEGIN FABLE-ENGINEER2 J3CUT (2026-09-25; lead GO to prepare for S25; APPROXIMATION): drop the birth update's gate-jet leg j = 3 (analyst's
# ablation survivor, fable_analyst_v1/RESULT.md sec. 36: raw +0.002% on torch VALID16). Full native saving (engineer2 census of S23 r100):
# -18.9 M flops per layer = -0.283 G/net = -0.091% bill, about -12 requests per layer. The direction stack shrinks to (3,n,n); directions
# 0..2 are formed, orthogonalized, weighted and summed exactly as before, so only the j3 term (c3 * orth(dir3) + X3 * dir3) leaves.
_E2_BD_STATE3 = {}   # three-direction twin of _E2_BD_STATE


def _e2_bd_state3(n):
    st = _E2_BD_STATE3.get(n)
    if st is None:
        D = fnp.empty((3, n, n), dtype=fnp.float64)
        st = dict(D=D, Dv=fnp.unstack(D), D12=D[1:3], diag=fnp.arange(3 * n) % n * (n + 1) + fnp.arange(3 * n) // n * (n * n),
                  old32=fnp.empty((n, n), dtype=fnp.float32))
        _E2_BD_STATE3[n] = st
    return st


'''
out = src.rstrip('\n') + '\n' + FRAG + b + '\n# END FABLE-ENGINEER2 J3CUT\n'
ast.parse(out)
open(out_path, 'w').write(out)
frag_path = out_path.replace('.py', '_fragment.py')
open(frag_path, 'w').write(FRAG + b + '\n# END FABLE-ENGINEER2 J3CUT\n')
print(out_path, hashlib.sha256(out.encode()).hexdigest()[:8], '| fragment', frag_path, hashlib.sha256(open(frag_path).read().encode()).hexdigest()[:8])
