#!/usr/bin/env python3
"""TAG fable-engineer (E-29): float64 -> float32 body patches, count-asserted exact-text replacements (refuses to overwrite <out>).
flopscope bills float64 at 2x float32. Patch sets (comma list):
  third  the third-order sums over the n x columns planes: the live diagonal sum(pp^3 lam) and its cache, omitted_radial_source's
         row_energy / column_sum matvecs, and the terminal k3 diagonal (plain form, or the physicist's L15 row-cut form when
         build_l15_rowcut.py ran first). pp and lam are already float32 (kernel outputs); only the reduced vectors widen to float64.
  bd     _bd_update's (4,n,n) direction stack in float32: projection, orthogonalization, coefficient sum and the new/old base
         products (the stored base is float32 anyway). The fills from the float64 K21/Cpre write straight into the float32 stack
         (billed as before); delta stays float64.
Also reachable as build_eng.py --f32 <sets> (applied to the base text before the kernel fragments), so porter's build_stack.sh
takes it after `--`, after the L15 row cut.  usage: build_f32.py <base.py> <out.py> <patches e.g. third or third,bd>"""
import sys, hashlib
from pathlib import Path


def apply(src, patches):
    s = [src]
    def rep(old, new, count=1):
        n = s[0].count(old)
        assert n == count, (n, count, old[:120])
        s[0] = s[0].replace(old, new)
    if "third" in patches:
        rep('''                _e2_pp64 = pp.astype(fnp.float64)
                _e2_pp2 = _e2_pp64 * _e2_pp64
                _e2_lam64 = lam.astype(fnp.float64)
                _E2_LAYER.update(pp=pp, lam=lam, pp64=_e2_pp64, pp2=_e2_pp2, lam64=_e2_lam64)
                diagonal = fnp.sum(_e2_pp2 * _e2_pp64 * _e2_lam64, axis=1)
                del _e2_pp64, _e2_pp2''',
            '''                _e2_pp2 = pp * pp   # F32 (E-29): third-order sums in float32 (pp, lam float32); the reduced vector widens to float64
                _E2_LAYER.update(pp=pp, lam=lam, pp64=pp, pp2=_e2_pp2, lam64=lam)
                diagonal = fnp.sum(_e2_pp2 * pp * lam, axis=1).astype(fnp.float64)
                del _e2_pp2''')
        rep('''    if L.get('pp') is pp and L.get('lam') is lam and 'pp64' in L:
        p, p2, lam64 = L.pop('pp64'), L.pop('pp2'), L['lam64']
    else:
        p = pp.astype(fnp.float64)
        p2 = p * p
        lam64 = lam.astype(fnp.float64)''',
            '''    if L.get('pp') is pp and L.get('lam') is lam and 'pp64' in L:
        p, p2, lam64 = L.pop('pp64'), L.pop('pp2'), L['lam64']   # F32 (E-29): float32 plane, its square and lam
    else:
        p = pp
        p2 = p * p
        lam64 = lam''')
        rep('''    row_energy = p2.T @ a
    column_sum = p.T @ beta
    contraction = fnp.sum(lam64 * row_energy * column_sum)''',
            '''    row_energy = (p2.T @ a.astype(fnp.float32)).astype(fnp.float64)   # F32 (E-29): float32 matvecs over the plane
    column_sum = (p.T @ beta.astype(fnp.float32)).astype(fnp.float64)
    contraction = fnp.sum(lam64.astype(fnp.float64) * row_energy * column_sum)''')
        plain = '''                k3 = fnp.sum(_cp_diagonal_cube(pp.astype(fnp.float64)) * lam.astype(fnp.float64), axis=1)
                k3 = k3 + (_flms_star_diagonal(w32, star) if _FULL_LOWER_MOMENTS'''
        rowcut = '''    k3_rows = fnp.sum(_cp_diagonal_cube(pp_rows.astype(fnp.float64)) * lam.astype(fnp.float64), axis=1)'''
        forms = (s[0].count(plain), s[0].count(rowcut))
        assert forms in ((1, 0), (0, 1)), ("terminal k3 form", forms)
        if forms == (1, 0):
            rep(plain, '''                k3 = fnp.sum(_cp_diagonal_cube(pp) * lam, axis=1).astype(fnp.float64)   # F32 (E-29)
                k3 = k3 + (_flms_star_diagonal(w32, star) if _FULL_LOWER_MOMENTS''')
        else:
            rep(rowcut, '''    k3_rows = fnp.sum(_cp_diagonal_cube(pp_rows) * lam, axis=1).astype(fnp.float64)   # F32 (E-29)''')
    if "bd" in patches:
        rep('''        D = fnp.empty((4, n, n), dtype=fnp.float64)
        st = dict(D=D, Dv=fnp.unstack(D), D12=D[1:3],''',
            '''        D = fnp.empty((4, n, n), dtype=fnp.float32)   # F32 (E-29): direction stack in float32
        st = dict(D=D, Dv=fnp.unstack(D), D12=D[1:3],''')
        rep('''    c = coefficients.astype(fnp.float64)
    alpha = aux['alpha']''', '''    c = coefficients.astype(fnp.float32)   # F32 (E-29)
    alpha = aux['alpha']''')
        rep('''    _e2_hold.append(fnp.multiply(st['D12'], eta, out=st['D12']))''',
            '''    _e2_hold.append(fnp.multiply(st['D12'], eta.astype(fnp.float32), out=st['D12']))''')
        rep('''    projection = fnp.sum(bp * D, axis=1) / denominator
    _e2_hold.append(fnp.subtract(D, bp * projection[:, None, :], out=D))''',
            '''    bp32 = bp.astype(fnp.float32)   # F32 (E-29): projection / orthogonalization / coefficient sum in float32
    projection = fnp.sum(bp32 * D, axis=1) / denominator.astype(fnp.float32)
    _e2_hold.append(fnp.subtract(D, bp32 * projection[:, None, :], out=D))''')
        rep('''    _e2_hold.append(fnp.multiply(bp, alive_column, out=old32))
    new32 = fnp.empty((n, n), dtype=fnp.float32)
    _e2_hold.append(fnp.multiply(bp + change, alive_column, out=new32))''',
            '''    _e2_hold.append(fnp.multiply(bp32, alive_column, out=old32))
    new32 = fnp.empty((n, n), dtype=fnp.float32)
    _e2_hold.append(fnp.multiply(bp32 + change, alive_column, out=new32))''')
    assert s[0] != src
    return s[0]


if __name__ == "__main__":
    base, out, patches = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
    src = base.read_text()
    s = apply(src, patches)
    assert not out.exists(), f"refusing to overwrite {out}"
    out.write_text(s)
    print(out, hashlib.sha256(s.encode()).hexdigest(), "base", hashlib.sha256(src.encode()).hexdigest()[:16], "patches", patches)
