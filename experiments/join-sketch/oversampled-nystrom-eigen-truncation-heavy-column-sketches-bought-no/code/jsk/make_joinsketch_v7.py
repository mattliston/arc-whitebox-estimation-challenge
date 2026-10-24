"""[opus-lead] JSK v7 = JSK-lite (v6: Nystrom sketch on the kc heaviest joined columns, core rescaled to the full trace) + the team's NT
join (opus_u64idea_v1/build_nt_v2.py: rank+p static rows in energy order, QR, then eigen-truncation back to rank: full eigh of the
(r+p) Nystrom core, or block q: keep the leading r-q frame columns and rotate the last q+p by the top-q eigenvectors of
M = CTT^2 + 2 CHT^T CHT). Downstream (Fj, overlap, factors, schedule, rank) unchanged. Patches BOTH live join bodies.
usage: make_joinsketch_v7.py <parent> <out> [kc json] [p json] [q]"""
import json, sys, hashlib
from pathlib import Path
parent, out = Path(sys.argv[1]), Path(sys.argv[2]); assert not out.exists()
kc = {int(k): int(v) for k, v in (json.loads(sys.argv[3]) if len(sys.argv) > 3 else {7: 2048, 10: 1536, 13: 1536}).items()}
P = {int(k): int(v) for k, v in (json.loads(sys.argv[4]) if len(sys.argv) > 4 else {7: 64, 10: 64, 13: 64}).items()}
Q = int(sys.argv[5]) if len(sys.argv) > 5 else 192
lines = parent.read_text().split('\n')

def find_block(start_idx, first, last):
    i = next(k for k in range(start_idx, len(lines)) if lines[k].startswith(first))
    j = next(k for k in range(i, len(lines)) if lines[k] == last)
    return i, j

NEW_SKETCH = '''        if _JOIN_NYSTROM:   # [opus-lead] JSK v7: Nystrom on the kc heaviest joined columns with rank+p static rows (NT), block eigen-truncation below; old-tier term and QR unchanged
            _jn_JJ = J * J
            _jn_d = fnp.matmul(_jn_JJ, weight)
            _jn_c = fnp.sum(_jn_JJ, axis=0) * weight
            del _jn_JJ
            if old_columns:
                _jn_d = _jn_d + fnp.sum((_shared_old384_mm(Qp, state['core'])) * Qp, axis=1)
            _jn_rows = (fnp.argsort(-(_jn_d.astype(fnp.float64)), kind='stable')[:rank + _nt_p] if _nt_p
                        else fnp.sort(fnp.argsort(-(_jn_d.astype(fnp.float64)), kind='stable')[:rank]))
            _jn_cols1 = fnp.argsort(-(_jn_c.astype(fnp.float64)), kind='stable')[:_JSK_KC[layer]]
            J1 = J[:, _jn_cols1]
            Y = _shared_old384_mm(J1, weight[_jn_cols1][:, None] * J1[_jn_rows, :].T)
            del J1, _jn_cols1, _jn_c
            if old_columns:
                Y = Y + (_shared_old384_mm(Qp, _shared_old384_mm(state['core'], (Qp[_jn_rows, :]).T)))
            del _jn_d'''

RESCALE_AFTER_FJ = '''        if _need_core:   # [opus-lead] JSK v7: the core (Nystrom or NT-truncated) is the subsample's Gram in the frame; rescale it to the full joined trace
            core = core * (fnp.sum(fnp.matmul(Fj * Fj, weight)) / fnp.trace(core))'''

FRAG = '''

# BEGIN OPUS-LEAD JSK v7 NT TRUNCATION (ported from opus_u64idea_v1/build_nt_v2.py; the print removed)
_NT_P = %r   # join layer -> extra static rows p (absent = plain rank-r Nystrom)
_NT_Q = %d   # 0 = full eigen truncation of the (r+p) core; q > 0 = block: keep the leading r-q frame, truncate the last q+p


def _nt_truncate(layer, rank, Q, R, rows, need_core):
    k = int(Q.shape[1])
    C = fnp.linalg.solve(Q[rows, :].astype(fnp.float32), R.astype(fnp.float32).T).astype(fnp.float32)   # Nystrom core in frame Q
    C = (C + C.T) * 0.5
    if not _NT_Q:
        ev, V = fnp.linalg.eigh(C)
        V = (V[:, k - rank:][:, ::-1]).copy()
        Qn = _shared_old384_mm(Q, V)
        core = fnp.diag((ev[k - rank:][::-1]).copy()).astype(fnp.float32) if need_core else None
    else:
        h = rank - _NT_Q
        CT = (C[:, h:]).copy()
        CTT = (CT[h:, :]).copy(); CHT = (CT[:h, :]).copy()
        M = fnp.matmul(CTT, CTT) + 2.0 * fnp.matmul(CHT.T, CHT)
        M = (M + M.T) * 0.5
        ev, V = fnp.linalg.eigh(M)
        X = (V[:, (k - h) - _NT_Q:][:, ::-1]).copy()
        Qn = fnp.concatenate([Q[:, :h], fnp.matmul(Q[:, h:], X)], axis=1)
        core = None
        if need_core:
            CX = fnp.matmul(CT, X)
            HX = (CX[:h, :]).copy()
            core = fnp.concatenate([fnp.concatenate([C[:h, :h], HX], axis=1),
                                    fnp.concatenate([HX.T, fnp.matmul(X.T, CX[h:, :])], axis=1)], axis=0).astype(fnp.float32)
            core = (core + core.T) * 0.5
    return Qn.astype(fnp.float32), core
# END OPUS-LEAD JSK v7 NT TRUNCATION
''' % (P, Q)

def patch_function(d):
    k = next(k for k in range(d, len(lines)) if lines[k].startswith('        Omega = w32[previous, :rank]'))
    lines[k + 1:k + 1] = ['        _nt_p = int(_NT_P.get(layer, 0)) if _JOIN_NYSTROM else 0   # [opus-lead] JSK v7: extra static rows at this join']
    i, j = find_block(d, '        if _JOIN_NYSTROM:   # PT-42 Nystrom join', '            del _jn_d')
    assert j - i == 8, (i, j)
    lines[i:j + 1] = NEW_SKETCH.split('\n')
    k = next(k for k in range(d, len(lines)) if lines[k].startswith('        if _need_core:   # opus-engineer a08'))
    lines[k:k + 1] = ['        if _nt_p:', '            Qn, core = _nt_truncate(layer, rank, Qn, unused_R, _jn_rows, _need_core)',
                      lines[k].replace('        if _need_core:', '        elif _need_core:', 1)]
    k = next(k for k in range(d, len(lines)) if lines[k].startswith('        Fj = _shared_old384_mm('))
    lines[k + 1:k + 1] = RESCALE_AFTER_FJ.split('\n')

ott = next(k for k in range(len(lines)) if lines[k] == '_OTT_ORIGINAL_PUSH = _shared_old384_push')
owner = max(k for k in range(ott) if lines[k].startswith('def _shared_old384_push('))
impl = [k for k in range(len(lines)) if lines[k].startswith('def _astra_bcc_push_impl(')]
assert len(impl) == 1 and impl[0] > owner
patch_function(impl[0]); patch_function(owner)
text = '\n'.join(lines)
assert text.count('JSK v7: Nystrom on the kc') == 2 and text.count('Qn, core = _nt_truncate(') == 2 and text.count('JSK v7: the core') == 2
text = text.replace('\n_JOIN_NYSTROM = True\n', '\n_JOIN_NYSTROM = True\n_JSK_KC = %r   # [opus-lead] JSK v7: join layer -> number of heaviest joined columns used for the Nystrom sketch\n' % (kc,), 1)
assert '_JSK_KC = ' in text
text = text + FRAG
out.write_text(text); compile(text, str(out), 'exec')
print('wrote', out, 'sha8', hashlib.sha256(text.encode()).hexdigest()[:8], 'kc', kc, 'p', P, 'q', Q)
