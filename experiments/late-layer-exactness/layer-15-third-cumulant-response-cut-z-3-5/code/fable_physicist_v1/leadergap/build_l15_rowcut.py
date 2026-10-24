#!/usr/bin/env python3
"""TAG fable-physicist (LG-4): L15 k3 row cut as a native build step (count-asserted text patch; refuses to overwrite).

At the terminal layer only nonlinear_scalar and the final hook consume k3. The CP and star parts of k3 are computed only on rows
R = select(zc - |mu/sigma|, 0): the units with |mu/sigma| <= zc, rounded up to a multiple of 32 by the card's own select, so every
product keeps its Winograd path. All other rows keep only the 3 v b term. The live _shared_old384_push is reused unchanged,
called with W restricted to the columns R, and the W5 port scope is suspended for that one call (the port requires the
registered 1024-column W; with no scope it dispatches the plain product). Works on the bank, e2 v7 and e2 v8 (_e2_H) forms,
before or after the physics, refit, kernel and join steps: it anchors on the push line and the two star-diagonal lines, which
are identical in all of them, and on the cube line in either form.
Torch reference: fable_physicist_v1/leadergap/arch_screen_v3.py cfg term32:<zc>.
usage: build_l15_rowcut.py <in.py> <out.py> [zc=3.5]"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
zc = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
if dst.exists():
    sys.exit(f'refusing to overwrite {dst}')
text = src.read_text()
if '_l15_rowcut_k3' in text:
    sys.exit('input already carries the L15 row cut')
lines = text.split('\n')

# the live _predict_binary is the last top-level definition
defs = [i for i, s in enumerate(lines) if s.startswith('def _predict_binary(')]
assert defs, 'no _predict_binary'
start = defs[-1]
end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(('def ', 'class '))), len(lines))
term = [i for i in range(start, end) if lines[i].strip() == 'if l == depth - 1 and final_hook is not None:']
assert len(term) == 1, f'expected one terminal block in the live _predict_binary, found {len(term)}'
push = [i for i in range(term[0], min(term[0] + 12, end))
        if lines[i].strip() == 'pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)']
assert len(push) == 1, f'expected one terminal push line, found {len(push)}'
i = push[0]
cube, star0, star1, bterm = (lines[i + k].strip() for k in (1, 2, 3, 4))
assert cube.startswith('k3 = fnp.sum(') and '_cp_diagonal_cube(' in cube and 'pp.astype(fnp.float64)' in cube, cube
assert star0 == 'k3 = k3 + (_flms_star_diagonal(w32, star) if _FULL_LOWER_MOMENTS', star0
assert star1 == 'else _star_compact_diagonal(w32, star, previous, previous_alive))', star1
assert bterm.startswith('k3 = k3 + ') and 'active_v' in bterm and 'bpre' in bterm, bterm
ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
lines[i:i + 4] = [ind + '# fable-physicist LG-4: L15 k3 row cut (CP + star parts only on rows with |mu/sigma| <= _L15_ROWCUT_Z).',
                  ind + 'k3 = _l15_rowcut_k3(l, mu, v, w32, p, previous, newborn, lam, shared_old, star)']

# functions the patch relies on must exist with the expected signatures
for need in ('def select(score, floor):', 'def _shared_old384_push(layer, w32, p, previous, newborn, lam, state):',
             'def _flms_star_diagonal(w, A):', 'def _cp_diagonal_cube(', '_W5_PORT_SCOPE = None'):
    assert need in text, f'missing {need!r}'

FUNC = f'''

# fable-physicist LG-4 (build_l15_rowcut.py): terminal k3 row cut, threshold on |mu/sigma|.
_L15_ROWCUT_Z = {zc!r}


def _l15_rowcut_k3(layer, mu, v, w32, p, previous, newborn, lam, state, star):
    global _W5_PORT_SCOPE
    if not _FULL_LOWER_MOMENTS:
        raise RuntimeError('L15 row cut expects the full lower-moment star diagonal')
    n = int(mu.shape[0])
    z = fnp.abs(mu / fnp.sqrt(fnp.maximum(v, 1e-10)))
    rows, unused_mask = select(_L15_ROWCUT_Z - z, 0.0)
    del unused_mask, z
    k3 = fnp.zeros(n, dtype=fnp.float64)
    if int(rows.shape[0]) == 0:
        return k3
    w_rows = w32[:, rows]
    saved_scope = _W5_PORT_SCOPE
    _W5_PORT_SCOPE = None
    try:
        pp_rows = _shared_old384_push(layer, w_rows, p, previous, newborn, lam, state)
    finally:
        _W5_PORT_SCOPE = saved_scope
    if tuple(pp_rows.shape) != (int(rows.shape[0]), int(lam.shape[0])):
        raise RuntimeError('L15 row cut: pushed roster shape')
    k3_rows = fnp.sum(_cp_diagonal_cube(pp_rows.astype(fnp.float64)) * lam.astype(fnp.float64), axis=1)
    del pp_rows
    k3_rows = k3_rows + _flms_star_diagonal(w_rows, star)
    fnp.put(k3, rows, k3_rows)
    return k3
'''
out = '\n'.join(lines) + FUNC
compile(out, str(dst), 'exec')
dst.write_text(out)
print(f'{src.name} -> {dst.name}: terminal push/cube/star lines {i + 1}-{i + 4} replaced; _L15_ROWCUT_Z = {zc}')
