#!/usr/bin/env python3
"""TAG fable-porter (PT-33): S19 = S18 + covhunter's learned-ON joint dials (fable_covhunter_v1/s6bcard_joint.py semantics, FIT from
s6bcard_jfitON1.py), as guarded native edits of the LIVE code; every dial off (--null) = the S18 op sequence (bit-identical, checked by
parity_layers.py). Edits (anchors are unique in the S18 file; each asserted):
  X1        _BD_J1_PHYS (j1 physical coefficient; S17/S18: 0.5)
  X0 X2 X3  _bd_update: physical raw (diag-zeroed, unorthogonalized) legs ADDED: change += sum_j x_j D_j, captured before the in-place
            orthogonalization (torch: change += raw * X_j after the learned c_j*orth(dir_j), j = 0, 2, 3)
  bscale    b *= bscale right after the b-fix line, before trace_post (torch b_post)
  s_carry/s_new  end-of-layer lam concatenate: [lam*s_carry, a*s_new, -(a*s_new)] (torch state_out, l <= 14)
  k4a/b/c   k4 *= band(l) (L1-7 / L8-11 / L12-15) right after k4 is formed, i.e. into the pair map (torch pair_in) and at L15 (k4_15)
  c0 c1     _K31R_TABLE (the pair map's K31 row factor, L1-14)
GN SEMANTICS (--gn=torch|shared): torch's s6bcard GN cov_pm hook recomputes C at L2-14 from the UNSCALED ctx['k4'] and a remainder31 with
the configure-time row constants 0.85/0.5. 'torch' mirrors the two cheap parts (_gn_dcov gets the unscaled k4 and a frozen 0.85/0.5 row
table) but NOT the main pair call's C(k4 scaled) vs C(k4 unscaled) difference; 'shared' = k4*band and the dialed table everywhere.
usage: build_s19_dials.py <in S18.py> <out.py> (--null | --fit=<json of the FIT dict>) [--gn=torch|shared]"""
import sys, json
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); args = sys.argv[3:]
null = '--null' in args
fit = None if null else json.loads(Path(next(a.split('=', 1)[1] for a in args if a.startswith('--fit='))).read_text())
gn = next((a.split('=', 1)[1] for a in args if a.startswith('--gn=')), 'torch'); assert gn in ('torch', 'shared')
s = src.read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
# _bd_update: raw legs captured before the in-place orthogonalization, added to the ordered sum
rep("    fnp.put(D, st['diag'], 0.0)\n",
    "    fnp.put(D, st['diag'], 0.0)\n"
    "    _s19_raw = None\n"
    "    if _S19_BD_ADD is not None:   # PT-33 S19: physical raw legs ADDED to the learned orthogonalized ones (covhunter X0/X2/X3)\n"
    "        for _s19_j, _s19_x in enumerate(_S19_BD_ADD):\n"
    "            if _s19_x != 0.0:\n"
    "                _s19_t = st['Dv'][_s19_j] * _s19_x\n"
    "                _s19_raw = _s19_t if _s19_raw is None else _s19_raw + _s19_t\n")
rep("    change = fnp.sum(D, axis=0)\n",
    "    change = fnp.sum(D, axis=0)\n"
    "    if _s19_raw is not None:\n"
    "        change = change + _s19_raw\n")
# b scale after the b-fix
rep("                b = b + _e2_H(_e2_H(_e2_H(_BFIX_GAMMA * _e2_H(radial_new - radial)) / 6) * m)\n",
    "                b = b + _e2_H(_e2_H(_e2_H(_BFIX_GAMMA * _e2_H(radial_new - radial)) / 6) * m)\n"
    "            if _S19_BSCALE is not None:   # PT-33 S19: carried b x bscale after the b-fix, before trace_post (covhunter; torch b_post)\n"
    "                b = b * _S19_BSCALE\n")
# lam: carried x s_carry, newborn x s_new
rep("            lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])\n",
    "            if _S19_LAM is None:\n"
    "                lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])\n"
    "            else:   # PT-33 S19: carried CP weights x s_carry, newborn x s_new (covhunter; torch state_out)\n"
    "                _s19_new = _e2_asel * _S19_LAM[1]\n"
    "                lam = fnp.concatenate([lam * _S19_LAM[0], _s19_new, -_s19_new])\n")
# k4 bands (pair map input, L15 terminal) + the GN dcov's k4
rep("            k4 = _e2_H(radial * vv) * vv\n",
    "            k4 = _e2_H(radial * vv) * vv\n"
    "            _s19_k4u = k4\n"
    "            if _S19_K4BAND is not None and l >= 1:   # PT-33 S19: per-unit k4 x band L1-7/8-11/12-15 (covhunter; torch pair_in, k4_15)\n"
    "                k4 = k4 * (_S19_K4BAND[0] if l <= 7 else (_S19_K4BAND[1] if l <= 11 else _S19_K4BAND[2]))\n")
rep("                C = C + _e2_H(_gn_dcov(l, mu, Cpre, K21, k4, aux))\n",
    "                C = C + _e2_H(_gn_dcov(l, mu, Cpre, K21, (_s19_k4u if _S19_GN_TORCH else k4), aux))\n")
rep("    r31 = _k31_row_scale(mu, k3, v) if '_k31_row_scale' in _e2_H(globals()) else None\n",
    "    r31 = ((_k31_row_scale_gn(mu, k3, v) if _S19_GN_TORCH else _k31_row_scale(mu, k3, v))\n"
    "           if '_k31_row_scale' in _e2_H(globals()) else None)   # PT-33 S19: torch GN remainder keeps the configure-time 0.85/0.5\n")
X = dict(X1=0.5, X0=0.0, X2=0.0, X3=0.0, bscale=1.0, s_carry=1.0, s_new=1.0, c0=0.85, c1=0.5, k4a=1.0, k4b=1.0, k4c=1.0) if null else fit
tail = ["", "", "# PT-33 S19 dials (covhunter learned-ON joint fit, s6bcard_joint semantics)" + (" -- NULL BUILD: every dial off (= S18)" if null else ""),
        f"_BD_J1_PHYS = {X['X1']!r}",
        "_S19_BD_ADD = " + (repr((X['X0'], 0.0, X['X2'], X['X3'])) if any(X[k] != 0.0 for k in ('X0', 'X2', 'X3')) else "None"),
        "_S19_BSCALE = " + (repr(X['bscale']) if X['bscale'] != 1.0 else "None"),
        "_S19_LAM = " + (repr((X['s_carry'], X['s_new'])) if (X['s_carry'], X['s_new']) != (1.0, 1.0) else "None"),
        "_S19_K4BAND = " + (repr((X['k4a'], X['k4b'], X['k4c'])) if (X['k4a'], X['k4b'], X['k4c']) != (1.0, 1.0, 1.0) else "None"),
        f"_S19_GN_TORCH = {(gn == 'torch' and not null)!r}",
        "_K31R_TABLE_GN = dict(_K31R_TABLE)   # the configure-time 0.85/0.5 table, read by _gn_dcov when _S19_GN_TORCH",
        "def _k31_row_scale_gn(mu, k3, v):",
        "    c = _K31R_TABLE_GN.get(_PHYS_LAYER[0])",
        "    if c is None:",
        "        return None",
        "    return c[0] + c[1] * mu * k3 / (v ** 2)"]
if not null and (X['c0'], X['c1']) != (0.85, 0.5):
    tail.append(f"_K31R_TABLE = {{l: ({X['c0']!r}, {X['c1']!r}) for l in _K31R_TABLE}}   # dialed pair-map row factor")
dst.write_text(s.rstrip('\n') + '\n' + '\n'.join(tail) + '\n')
print(f"wrote {dst} ({'NULL' if null else 'FIT'}, gn={gn})")
