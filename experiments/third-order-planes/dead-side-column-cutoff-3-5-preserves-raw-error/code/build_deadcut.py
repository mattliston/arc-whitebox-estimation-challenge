#!/usr/bin/env python3
"""TAG fable-porter (PT-15): native DEAD-SIDE ROW CUT at L1-14 (flag --deadcut35; analyst AN-50 variant (a), E-48 2b). Count-asserted
edits of a finished candidate (e.g. S11 acf67b9d or S11 at --rank=R):
(1) the LIVE (last) _shared_old384_push gains `rows=None`: its join (range finder sketch Omega = W[prev, :rank], QR, factors, core) keeps
    the FULL W; only the projection (U = W[prev, rows]^T basis, the old-bank rebuild U F) and the young/newborn push use wx = W[:, rows];
    rows=None is the unchanged code path (same objects, same ops);
(2) the live loop's push call becomes _deadcut_push(..., _e2_a): at L1-14 kept rows = select(a + 3.5, 0) (the L15 row-cut rule: rows with
    a = mu/sigma >= -3.5, count rounded up to 32 by descending a); if a row is dropped, the push runs on the kept rows (W5 port off for that
    call, as the L15 cut does) and the pushed columns and the frame pre-basis are gathered back to all rows with exact zeros on dropped rows.
Semantics (torch rowdrop_s6b.py mode a masks p64 = float64 pushed columns right after the push): dropped rows feed only their own CP k3
diagonal (-> 0, as the mask), the omitted-radial source (row weights masked by omitted/alive: unchanged) and the carry p = (gate*alive)*pp
(0 either way); the read uses response/live rows and U[live]. --null writes _DEADCUT_LAYERS = (1, 0): every layer takes the old path.
Options (PT-15b, 2026-09-24): --z=Z threshold (default 3.5: rows with a >= -Z kept); --round=R kept-row count rounded up to R (default 32 =
the card's select, byte-identical to build_deadcut_v1.py; 64 = one local stable selection so a kept push has m % 64 == 0 for the six-level
padded kernel). usage: build_deadcut.py <in.py> <out.py> [--null] [--z=Z] [--round=R]"""
import sys, hashlib
from pathlib import Path
src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]); null = '--null' in sys.argv[3:]
Z = next((a.split('=', 1)[1] for a in sys.argv[3:] if a.startswith('--z=')), '3.5')
RND = int(next((a.split('=', 1)[1] for a in sys.argv[3:] if a.startswith('--round=')), '32'))
assert RND in (32, 64, 128) and 0 < float(Z) < 10, (Z, RND)
assert not out_path.exists(), f'refusing to overwrite {out_path}'
src = src_path.read_text()
i = src.rfind('\ndef _shared_old384_push('); j = src.find('\ndef ', i + 5); assert i > 0 and j > i
fn = src[i:j]
def sub(text, old, new, n=1):
    assert text.count(old) == n, ('anchor', old[:70], text.count(old)); return text.replace(old, new)
fn = sub(fn, "def _shared_old384_push(layer, w32, p, previous, newborn, lam, state):",
             "def _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):")
fn = sub(fn, "return _push_bank_shared_newborn(w32, p, previous, newborn)",
             "return _push_bank_shared_newborn(w32 if rows is None else w32[:, rows], p, previous, newborn)")
lines = fn.split('\n'); k = [n for n, l in enumerate(lines) if "'shared_basis'" in l]; assert len(k) == 1, k
u = lines[k[0]]; assert u.count('w32[previous, :]') == 1, u
indent = u[:len(u) - len(u.lstrip())]
lines[k[0]] = u.replace('w32[previous, :]', 'wx[previous, :]')
lines.insert(k[0], indent + 'wx = w32 if rows is None else w32[:, rows]   # PT-15 deadcut: projection rows only; the join above keeps the full W')
fn = '\n'.join(lines)
fn = sub(fn, "young = _push_bank_shared_newborn(w32, ", "young = _push_bank_shared_newborn(wx, ")
out = src[:i] + fn + src[j:]
k = out.rfind('\ndef _predict_binary(')
out = out[:k] + sub(out[k:], "pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)",
                    "pp = _deadcut_push(l, w32, p, previous, newborn, lam, shared_old, _e2_a)")
out = out.rstrip('\n') + '\n' + '''

# BEGIN FABLE-PORTER DEAD-SIDE ROW CUT (PT-15 --deadcut35, 2026-09-24; see build_deadcut.py). Kept rows at L1-14 = select(a + 3.5, 0).
_DEADCUT_Z = %s
_DEADCUT_LAYERS = %s


def _deadcut_push(layer, w32, p, previous, newborn, lam, state, a):
    global _W5_PORT_SCOPE
    if not _DEADCUT_LAYERS[0] <= layer <= _DEADCUT_LAYERS[1]:
        return _shared_old384_push(layer, w32, p, previous, newborn, lam, state)
    n = int(a.shape[0])
%s    kept = int(rows.shape[0])
    if kept >= n:
        return _shared_old384_push(layer, w32, p, previous, newborn, lam, state)
    saved_scope = _W5_PORT_SCOPE
    _W5_PORT_SCOPE = None
    try:
        pp_rows = _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=rows)
    finally:
        _W5_PORT_SCOPE = saved_scope
    if int(pp_rows.shape[0]) != kept:
        raise RuntimeError('deadcut: pushed row count')
    inverse = fnp.full(n, kept, dtype=fnp.int64)
    fnp.put(inverse, rows, fnp.arange(kept, dtype=fnp.int64))
    pp = fnp.concatenate([pp_rows, fnp.zeros((1, int(pp_rows.shape[1])), dtype=pp_rows.dtype)], axis=0)[inverse]
    del pp_rows
    basis = state.get('pre_basis')
    if basis is not None:
        if int(basis.shape[0]) != kept:
            raise RuntimeError('deadcut: frame pre-basis rows')
        state['pre_basis'] = fnp.concatenate([basis, fnp.zeros((1, int(basis.shape[1])), dtype=basis.dtype)], axis=0)[inverse]
    return pp
# END FABLE-PORTER DEAD-SIDE ROW CUT
''' % (Z, '(1, 0)   # NULL build: no layer is cut' if null else '(1, 14)',
       '    rows, unused_mask = select(a + _DEADCUT_Z, 0.0)\n    del unused_mask\n' if RND == 32 else
       ('    score = a + _DEADCUT_Z   # kept count rounded up to %d (select rounds to 32): m %% 64 == 0 for the six-level padded push\n'
        '    count = min(n, (int(fnp.nonzero(score >= 0.0)[0].shape[0]) + %d) // %d * %d)\n'
        '    rows = fnp.argsort(-score, kind=\'stable\')[:count]\n') % (RND, RND - 1, RND, RND))
out_path.write_text(out)
print(dict(out=str(out_path), null=null, z=Z, round=RND, sha256=hashlib.sha256(out.encode()).hexdigest(), base=hashlib.sha256(src.encode()).hexdigest()[:16]))
