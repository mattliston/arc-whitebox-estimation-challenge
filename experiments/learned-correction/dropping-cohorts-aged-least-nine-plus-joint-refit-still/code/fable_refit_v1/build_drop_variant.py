#!/usr/bin/env python3
"""TAG fable-refit (RF-3): build cand_drop<A>.py from the 331824 candidate: cohorts of the shared old tier are DROPPED (columns removed
from F / the lazy groups, lam and the cohort table compacted, core decremented) once their age reaches A (A >= 6). The joining cohort
is extracted before the drop; the caller's lam is replaced by state['lam_out'] right after the push. A=16 = no drops (must reproduce
the parent).  usage: build_drop_variant.py <candidate.py> <A> <out.py>"""
import sys, re
from pathlib import Path
cand, A, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
src = cand.read_text()
# 1) state: add drop bookkeeping (last definition)
i = src.rfind("def _shared_old384_state():\n")
j = src.index("\n\n\n", i)
state_new = ("def _shared_old384_state():\n"
             "    return dict(columns=0, next_column=0, cohorts=[], joins=[],\n"
             "                Q=None, F=None, core=None, pre_basis=None, lazy=None, dropped=0, lam_out=None)\n")
src = src[:i] + state_new + src[j:]
# 2) push: insert the drop block (last definition)
i = src.rfind("def _shared_old384_push(layer, w32, p, previous, newborn, lam, state):\n")
j = src.index("\n\n\ndef ", i)
push = src[i:j]
def rep(s, old, new):
    assert s.count(old) == 1, (old, s.count(old)); return s.replace(old, new)
# early layers: still record lam_out
push = rep(push, "        return _push_bank_shared_newborn(w32, p, previous, newborn)\n",
                 "        state['lam_out'] = lam\n        return _push_bank_shared_newborn(w32, p, previous, newborn)\n")
# join branch: after extracting the joining cohort J and its coefficients, drop the oldest cohort if it reached age A
push = rep(push, "        coefficient = lam[start:stop]\n        weight = coefficient * coefficient\n",
                 "        coefficient = lam[start:stop]\n        weight = coefficient * coefficient\n"
                 "        old_columns_pre = old_columns\n"
                 "        lam, old_columns = _shared_old384_drop(layer, lam, state)\n"
                 "        Qp = state['Q'][previous, :] if old_columns else None\n")
# the range finder / core update use old_columns (post-drop) and Qp: unchanged text. The join's own bookkeeping:
push = rep(push, "        old_columns = stop\n        state['columns'] = old_columns\n",
                 "        old_columns = old_columns + width\n        state['columns'] = old_columns\n"
                 "        old_columns_p = stop\n")
push = rep(push, "        if len(state['joins']) != 10 or not old_columns:\n            raise RuntimeError('Shared-old384 terminal join history')\n        active_basis = Qp\n",
                 "        if len(state['joins']) != 10 or not old_columns:\n            raise RuntimeError('Shared-old384 terminal join history')\n"
                 "        old_columns_p = old_columns\n        lam, old_columns = _shared_old384_drop(layer, lam, state)\n"
                 "        active_basis = state['Q'][previous, :]\n")
push = rep(push, "    if old_columns > int(inherited_columns):\n        raise RuntimeError('Shared-old384 attempted newborn confinement')\n"
                 "    young_newborn = (int(inherited_columns) - old_columns, selected, base, t, s)\n"
                 "    young = _push_bank_shared_newborn(w32, p[:, old_columns:], previous, young_newborn)\n",
                 "    if old_columns_p > int(inherited_columns):\n        raise RuntimeError('Shared-old384 attempted newborn confinement')\n"
                 "    young_newborn = (int(inherited_columns) - old_columns_p, selected, base, t, s)\n"
                 "    young = _push_bank_shared_newborn(w32, p[:, old_columns_p:], previous, young_newborn)\n"
                 "    state['lam_out'] = lam\n")
drop_fn = '''

_DROP_AGE = %d
def _shared_old384_drop(layer, lam, state):
    """Drop the oldest live cohort once its age reaches _DROP_AGE: remove its columns from the compact roster (they are always the
    first block of the old tier), from F or the open lazy group, from the core, and from lam; shift the cohort table."""
    b = layer - _DROP_AGE
    if b < 0 or state['dropped'] > b:
        return lam, int(state['columns'])
    if state['dropped'] != b:
        raise RuntimeError('Shared-old384 drop order')
    start, wb = state['cohorts'][b]
    if wb <= 0:
        state['dropped'] = b + 1
        return lam, int(state['columns'])
    if start != 0 or wb > int(state['columns']):
        raise RuntimeError('Shared-old384 dropped cohort must lead the roster')
    group = state['lazy']
    if group is None:
        F = state['F']
        Fb = F[:, :wb]
        state['F'] = F[:, wb:]
    else:
        base = group['base']
        Fb = _shared_old384_mm(group['map'], base[:, :wb])
        group['base'] = base[:, wb:]
        group['prefix'] = int(group['prefix']) - wb
        if group['prefix'] == 0:
            # the whole unrotated prefix is gone: the recent cohort is already in the current frame
            state['F'] = group['join']
            state['lazy'] = None
    wgt = lam[:wb] * lam[:wb]
    state['core'] = state['core'] - _shared_old384_mm(Fb, wgt[:, None] * Fb.T)
    state['columns'] = int(state['columns']) - wb
    state['next_column'] = int(state['next_column']) - wb
    state['cohorts'] = [(0, 0) if c == b else ((s - wb, w) if c > b else (s, w)) for c, (s, w) in enumerate(state['cohorts'])]
    state['dropped'] = b + 1
    return lam[wb:], int(state['columns'])
''' % A
src = src[:i] + push + drop_fn + src[j:]
# 2b) _slr_join_factors: allow the no-group path at any layer (a drop can dissolve the lazy group)
a1 = "    if layer in _SLR_CONSOLIDATE_LAYERS:\n        _slr_require(group is not None and group['layer'] == layer - 1,\n                     'immediate next-join consolidation')\n"
b1 = "    if layer in _SLR_CONSOLIDATE_LAYERS and group is not None:\n        _slr_require(group['layer'] == layer - 1,\n                     'immediate next-join consolidation')\n"
assert src.count(a1) == 1; src = src.replace(a1, b1)
a2 = "    _slr_require(layer in (6, 7) and group is None,\n                 'all other old-factor joins retain their original schedule')\n"
b2 = "    _slr_require(group is None,\n                 'all other old-factor joins retain their original schedule')\n"
assert src.count(a2) == 1; src = src.replace(a2, b2)
# 3) the live loop: take the compacted lam after the push (last occurrence)
H = "            pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)\n"
k = src.rfind(H); assert k > 0
src = src[:k] + H + "            lam = shared_old['lam_out']\n" + src[k + len(H):]
H2 = "                pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)\n"
k2 = src.rfind(H2); assert k2 > 0
src = src[:k2] + H2 + "                lam = shared_old['lam_out']\n" + src[k2 + len(H2):]
out.write_text(src); print("wrote", out, len(src))
