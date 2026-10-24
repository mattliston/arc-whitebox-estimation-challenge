#!/usr/bin/env python3
"""TAG fable-engineer (E-48 --colpack, 2026-09-24): the exact per-column package on a post-diet candidate (run it AFTER
fable_engineer2_v1/build_e2mem.py: its loop edits sit on lines the diet rewrites). Bit-identical predictions; the bill falls only by
the work removed. Refuses to overwrite; every edit is count-asserted on the input's LIVE definitions.
  E1  a six-level push wider than one 4096-column packet transforms its left operand ONCE. The left side's levels 5 and 6 are bound
      in a side buffer (34.4M floats = 138 MB, allocated once per process) that the first packet's unwind cannot overwrite; the
      second packet skips the left transform and reads the same values. Covers the young pushes and the old-tier rebuild products.
  E3  pp*pp is formed before the read, and the read takes its squared response rows from it (the same products) instead of
      squaring the gathered rows again.
  E4  the gated young columns p are kept only on the rows the next push reads (this layer's live rows), formed from the read's
      already-gathered rows: no gating of all 1024 rows, no row gathers of p at the next push, at joins and at L15.
usage: build_colpack.py <in.py> <out.py>"""
import sys, hashlib
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists():
    sys.exit(f"refusing to overwrite {dst}")
s = src.read_text()

def live_def(text, name):
    i = text.rfind("\ndef " + name + "(")
    assert i >= 0, name
    lines = text[i + 1:].split("\n"); out = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() == "" or ln.startswith((" ", "\t")):
            out.append(ln)
        else:
            break
    while out and out[-1].strip() == "":
        out.pop()
    return i + 1, "\n".join(out)

def rep(text, old, new, where):
    n = text.count(old)
    assert n == 1, (where, old[:90], n)
    return text.replace(old, new)

for name in ("_packed_product_4", "_push_bank_shared_newborn", "_shared_old384_push"):   # redefined here: no captured alias may bypass them
    assert s.count("= " + name + "\n") == 0 and s.count("=" + name + "\n") == 0, ("captured alias", name)
assert "E-48 --colpack" not in s, "input already has --colpack"

# ---- loop edits (E3, E4) in the live _predict_binary
pi, pb = live_def(s, "_predict_binary")
OLD_PP2 = "                _e2_pp2 = pp * pp   # F32 (E-29): third-order sums in float32 (pp, lam float32); the reduced vector widens to float64\n"
nb = rep(pb + "\n", OLD_PP2, "", "late pp2")
nb = rep(nb,
    "                left = pp[response_indices]\n"
    "                right = pp[live_indices, int(shared_old['columns']):]   # E2-R: only the young columns of right are ever read\n"
    "                response_plane = _e2_H(_e2_shared_old384_read_young(left, right, lam, live_indices, shared_old)).astype(fnp.float64)\n"
    "                del left, right   # E2-M: dead after the read (they lived to the next layer, through the live peak)\n",
    OLD_PP2.rstrip("\n") + ". E-48 --colpack E3: formed before the read, which takes its squared response rows from it\n"
    "                left2 = _e2_pp2[response_indices]\n"
    "                right = pp[live_indices, int(shared_old['columns']):]   # E2-R: only the young columns of right are ever read\n"
    "                response_plane = _e2_H(_cp_read_young_sq(left2, right, lam, live_indices, shared_old)).astype(fnp.float64)\n"
    "                del left2   # E-48 --colpack E4: right stays until the end-of-layer gating, which forms the compact p from it\n",
    "read block")
nb = rep(nb,
    "            p = fnp.einsum('i,ij->ij', _e2_ga, pp, optimize=False)\n",
    "            p = fnp.einsum('i,ij->ij', _e2_ga[live_indices], right if len(lam) else pp[live_indices, int(shared_old['columns']):], optimize=False)   # E-48 --colpack E4: the gated young columns on the rows the next push reads\n"
    "            if len(lam):\n"
    "                del right\n",
    "gating")
s = s[:pi] + nb.rstrip("\n") + s[pi + len(pb):]

# ---- generated functions (E3: squared read and inner; E4: compact-p pushes), from the input's live sources
_, rd = live_def(s, "_e2_shared_old384_read_young")
rd = rep(rd, "def _e2_shared_old384_read_young(left, young_right, lam, live_indices, state):", "def _cp_read_young_sq(left, young_right, lam, live_indices, state):\n    # E-48 --colpack E3: = _e2_shared_old384_read_young with left already squared ((pp*pp)[response rows]); the same products", "read def")
rd = rep(rd, "return _mm(left * left, (young_right * lam).T)", "return _mm(left, (young_right * lam).T)", "read nocols")
rd = rep(rd, "inner = _slr_inner(old_left, lam[:old_columns], state)", "inner = _cp_slr_inner_sq(old_left, lam[:old_columns], state)", "read inner")
rd = rep(rd, "young_plane = _mm(young_left * young_left, (young_right * lam[old_columns:]).T)", "young_plane = _mm(young_left, (young_right * lam[old_columns:]).T)", "read young")
_, sin = live_def(s, "_slr_inner")
sin = rep(sin, "def _slr_inner(old_left, lam_old, state):", "def _cp_slr_inner_sq(old_left, lam_old, state):\n    # E-48 --colpack E3: = _slr_inner with old_left already squared; the same products", "inner def")
sin = rep(sin, "return _shared_old384_mm(_e2_H(old_left * old_left), _e2_H(weighted_factors.T))", "return _shared_old384_mm(_e2_H(old_left), _e2_H(weighted_factors.T))", "inner flat")
sin = rep(sin, "inner0 = _shared_old384_mm(_e2_H(left0 * left0), _e2_H(weights0.T))", "inner0 = _shared_old384_mm(_e2_H(left0), _e2_H(weights0.T))", "inner lazy0")
sin = rep(sin, "inner1 = _shared_old384_mm(_e2_H(left1 * left1), _e2_H(weights1.T))", "inner1 = _shared_old384_mm(_e2_H(left1), _e2_H(weights1.T))", "inner lazy1")
_, nbp = live_def(s, "_push_bank_shared_newborn")
nbp = rep(nbp, "def _push_bank_shared_newborn(w,p,previous,newborn):", "def _push_bank_shared_newborn(w,p,previous,newborn):\n    # E-48 --colpack E4: p is compact (its rows are previous), so no row gathers; the same products", "nb def")
nbp = rep(nbp, "return _mm(wp, _e2_H(p[previous, :]))", "return _mm(wp, _e2_H(p))", "nb none")
nbp = rep(nbp, "inherited = _mm(wp, _e2_H(p[previous, :old_columns]))", "inherited = _mm(wp, _e2_H(p[:, :old_columns]))", "nb inherited")
_, sop = live_def(s, "_shared_old384_push")
sop = rep(sop, "    old_columns = int(state['columns'])\n", "    old_columns = int(state['columns'])\n    _cp_old0 = old_columns   # E-48 --colpack E4: p holds the roster columns from here on, on the rows previous\n", "sop old0")
sop = rep(sop, "if start != old_columns or width <= 0 or stop > int(p.shape[1]):", "if start != old_columns or width <= 0 or stop - _cp_old0 > int(p.shape[1]):", "sop check")
sop = rep(sop, "J = p[previous, start:stop]", "J = p[:, start - _cp_old0:stop - _cp_old0]", "sop J")
sop = rep(sop, "young = _push_bank_shared_newborn(w32, _e2_H(p[:, old_columns:]), previous, young_newborn)", "young = _push_bank_shared_newborn(w32, _e2_H(p[:, old_columns - _cp_old0:]), previous, young_newborn)", "sop young")
assert "p[previous" not in sop and "p[previous" not in nbp, "a row gather of p is left"

frag = (Path(__file__).resolve().parent / "colpack_fragment_e1.py").read_text()
s = s.rstrip("\n") + "\n\n\n# BEGIN FABLE-ENGINEER E-48 --colpack (2026-09-24; bit-identical): generated from this file's live definitions by build_colpack.py\n" \
    + rd + "\n\n\n" + sin + "\n\n\n" + nbp + "\n\n\n" + sop + "\n\n\n" + frag.rstrip("\n") + "\n# END FABLE-ENGINEER E-48 --colpack\n"
compile(s, str(dst), "exec")
dst.write_text(s)
print(f"{hashlib.sha256(s.encode()).hexdigest()} {dst.name}  (input {hashlib.sha256(src.read_bytes()).hexdigest()[:16]})")
