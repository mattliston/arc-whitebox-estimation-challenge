#!/usr/bin/env python3
"""TAG fable-join (JS-1): native JOIN SCHEDULE for the shared old tier. Exact string edits (count-asserted) of the LAST definition of
_shared_old384_push in any base that carries the bank's version (bank, build_e2, engineer v9, build_physics outputs).
Cohorts join the fixed rank-384 frame only at the layers of _JOIN_SCHEDULE (several cohorts at once); between joins the frame is only
pushed (no range finder / QR / projection / factor rotation). 'every' = the bank's schedule (must be bit-exact).
usage: build_join_schedule.py <base.py> <out.py> <every|even|odd>"""
import sys, hashlib
from pathlib import Path
SCHED = {
    'every': {l: (l - 5, l - 5) for l in range(5, 15)},
    'even': {6: (0, 1), 8: (2, 3), 10: (4, 5), 12: (6, 7), 14: (8, 9)},
    'odd': {5: (0, 0), 7: (1, 2), 9: (3, 4), 11: (5, 6), 13: (7, 8)},
    'every3': {7: (0, 2), 10: (3, 5), 13: (6, 8)},
    'late2': {5: (0, 0), 6: (1, 1), 7: (2, 2), 8: (3, 3), 10: (4, 5), 12: (6, 7), 14: (8, 9)},
    'every3b': {8: (0, 2), 11: (3, 5), 14: (6, 8)},
    'every3c': {6: (0, 1), 9: (2, 4), 12: (5, 7)},
}
base, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
src = base.read_text()
i = src.rfind('\ndef _shared_old384_push(')
j = src.find('\ndef ', i + 5)
assert i > 0 and j > i
fn = src[i:j]
def sub(old, new, text):
    assert text.count(old) == 1, ('anchor', old[:60], text.count(old))
    return text.replace(old, new)
def sub_any(pairs, text):
    # first (old, new) pair whose anchor occurs exactly once (plain bank form, then engineer2's _e2_H hold form)
    for old, new in pairs:
        if text.count(old) == 1:
            return text.replace(old, new)
    raise AssertionError(('no anchor form matched', pairs[0][0][:60]))
fn = sub("""    if layer < 5:
        if state['columns'] or state['joins']:
            raise RuntimeError('Shared-old384 premature confinement')
        return _push_bank_shared_newborn(w32, p, previous, newborn)
""", """    joins_here = _JOIN_SCHEDULE.get(layer)
    if not state['columns'] and joins_here is None:
        if state['joins']:
            raise RuntimeError('Shared-old384 premature confinement')
        return _push_bank_shared_newborn(w32, p, previous, newborn)
""", fn)
fn = sub("""    if layer < 15:
        joining = layer - 5
        if len(state['joins']) != joining:
            raise RuntimeError('Shared-old384 fixed join order')
        start, width = state['cohorts'][joining]
        stop = start + width
""", """    if joins_here is not None:
        joining = int(joins_here[0])
        start = int(state['cohorts'][joins_here[0]][0])
        stop = int(state['cohorts'][joins_here[1]][0]) + int(state['cohorts'][joins_here[1]][1])
        width = stop - start
""", fn)
fn = sub_any([("""        Y = _shared_old384_mm(J, weight[:, None] * _shared_old384_mm(J.T, Omega))
""", """        Y = _shared_old384_mm(J, weight[:, None] * _join_rows_mm(J.T, Omega))
"""), ("_e2_H(_shared_old384_mm(_e2_H(J.T), Omega))", "_e2_H(_join_rows_mm(_e2_H(J.T), Omega))")], fn)
fn = sub("""        if len(state['joins']) != 10 or not old_columns:
            raise RuntimeError('Shared-old384 terminal join history')
""", """        if len(state['joins']) != len(_JOIN_SCHEDULE) or not old_columns:
            raise RuntimeError('Shared-old384 terminal join history')
""", fn)
helper = '''

_JOIN_SCHEDULE = %r


def _join_rows_mm(left, right):
    """Row-tiled product for a joined block wider than one signed cohort (tiles of <= 1024 rows; the counted concatenate restores
    order). Blocks of <= 2048 rows take exactly the original _shared_old384_mm call (bit-identical); wider blocks are tiled into
    _mm calls (the same kernel) of <= 1024 rows."""
    rows = int(left.shape[0])
    if rows <= 2048:
        return _shared_old384_mm(left, right)
    return fnp.concatenate([_mm(left[start:min(start + 1024, rows)], right) for start in range(0, rows, 1024)], axis=0)
''' % (SCHED[name],)
new = src[:i] + fn + src[j:]
if name != 'every':
    # Re-time the bank's lazy factor rotation (_slr_*) to the new join layers: the first join that has old factors materializes,
    # later joins alternate lazy / consolidate, the last join stays lazy (as in the bank, never consolidated).
    joins_old = sorted(SCHED[name])[1:]
    mat, lazy, cons, prev = [joins_old[0]], [], [], {}
    for k, L in enumerate(joins_old[1:]):
        if k % 2 == 0: lazy.append(L)
        else: cons.append(L); prev[L] = joins_old[joins_old.index(L) - 1]
    new = sub("_SLR_LAZY_LAYERS = (8, 10, 12, 14)\n_SLR_CONSOLIDATE_LAYERS = (9, 11, 13)\n",
              "_SLR_LAZY_LAYERS = %r\n_SLR_CONSOLIDATE_LAYERS = %r\n_SLR_MATERIALIZE_LAYERS = %r\n_SLR_PREVIOUS_JOIN = %r\n"
              % (tuple(lazy), tuple(cons), tuple(mat), prev), new)
    new = sub_any([("        _slr_require(group is not None and group['layer'] == layer - 1,\n",
                    "        _slr_require(group is not None and group['layer'] == _SLR_PREVIOUS_JOIN[layer],\n"),
                   ("group['layer'] == _e2_H(layer - 1)", "group['layer'] == _SLR_PREVIOUS_JOIN[layer]")], new)
    new = sub("    _slr_require(layer in (6, 7) and group is None,\n",
              "    _slr_require(layer in _SLR_MATERIALIZE_LAYERS and group is None,\n", new)
new = new.rstrip('\n') + '\n' + helper
out.write_text(new)
print(dict(out=str(out), schedule=name, sha256=hashlib.sha256(new.encode()).hexdigest()[:16], edits=4))
