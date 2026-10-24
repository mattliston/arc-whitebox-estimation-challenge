"""[opus-lead] JSK v1: build <out> from <parent> by replacing the Nystrom join sketch (Y = G[:, rows], G = J W J^T + Qp core Qp^T, rows =
the rank largest Gram diagonals) in BOTH live join functions (_shared_old384_push fallback and _astra_bcc_push_impl) with a
column-subsampled power-step sketch: Y1 = G0[:, rows] on the kc0 heaviest joined columns (plus the exact old-tier term), Y = G1 Y1 on
the kc heaviest columns (plus the old-tier term), QR as before; the Nystrom core (a solve) becomes the exact Gram core of the kc
columns in the new frame, Fj1 W1 Fj1^T. Plan per join layer: _JSK_PLAN = {layer: (kc0, kc)}. Offline basis: joinsketch/basis35_v*.txt.
usage: make_joinsketch_v1.py <parent.py> <out.py> [plan json]"""
import json, sys, re, hashlib
from pathlib import Path
parent, out = Path(sys.argv[1]), Path(sys.argv[2]); assert not out.exists()
plan = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {7: (1024, 2048), 10: (1024, 1536), 13: (768, 1536)}
plan = {int(k): tuple(int(x) for x in v) for k, v in (plan.items() if isinstance(plan, dict) else plan)}
src = parent.read_text()
lines = src.split('\n')

def find_block(start_idx, first, last):
    i = next(k for k in range(start_idx, len(lines)) if lines[k].startswith(first))
    j = next(k for k in range(i, len(lines)) if lines[k] == last)
    return i, j

NEW_SKETCH = '''        if _JOIN_NYSTROM:   # [opus-lead] JSK v1: column-subsampled Gram power step (jsk/JOINSKETCH_v1.md): start Y1 = G0[:, rows] on the kc0 heaviest joined columns, then Y = G1 Y1 on the kc heaviest; the old tier enters exactly through its core
            _jn_JJ = J * J
            _jn_d = fnp.matmul(_jn_JJ, weight)
            _jn_c = fnp.sum(_jn_JJ, axis=0) * weight
            del _jn_JJ
            if old_columns:
                _jn_Qc = _shared_old384_mm(Qp, state['core'])
                _jn_d = _jn_d + fnp.sum(_jn_Qc * Qp, axis=1)
            _jn_rows = fnp.sort(fnp.argsort(-(_jn_d.astype(fnp.float64)), kind='stable')[:rank])
            _jn_kc0, _jn_kc = _JSK_PLAN[layer]
            _jn_order = fnp.argsort(-(_jn_c.astype(fnp.float64)), kind='stable')
            _jn_cols1 = fnp.sort(_jn_order[:_jn_kc])
            _jn_cols0 = fnp.sort(_jn_order[:_jn_kc0])
            J1 = J[:, _jn_cols1]
            J0 = J[:, _jn_cols0]
            Y = _shared_old384_mm(J0, weight[_jn_cols0][:, None] * J0[_jn_rows, :].T)
            if old_columns:
                Y = Y + _shared_old384_mm(_jn_Qc, Qp[_jn_rows, :].T)
            _jn_Z = _shared_old384_mm(J1.T, Y)
            _jn_Y2 = _shared_old384_mm(J1, weight[_jn_cols1][:, None] * _jn_Z)
            if old_columns:
                _jn_Y2 = _jn_Y2 + _shared_old384_mm(_jn_Qc, _shared_old384_mm(Qp.T, Y))
                del _jn_Qc
            Y = _jn_Y2
            del _jn_d, _jn_c, _jn_order, _jn_cols0, J0, _jn_Z, _jn_Y2'''

NEW_CORE_AFTER_FJ = '''        if _need_core:   # [opus-lead] JSK v1: exact Gram core of the kc heaviest joined columns in the new frame (PSD), replacing the Nystrom solve
            _jn_F1 = Fj[:, _jn_cols1]
            core = _shared_old384_mm(_jn_F1 * weight[_jn_cols1][None, :], _jn_F1.T)
            core = ((core + core.T) * 0.5).astype(fnp.float32)
            del _jn_F1
        del J1, _jn_cols1'''

def patch_function(d):
    # 1. sketch block
    i, j = find_block(d, '        if _JOIN_NYSTROM:   # PT-42 Nystrom join', '            del _jn_d')
    assert j - i == 8, (i, j)
    lines[i:j + 1] = NEW_SKETCH.split('\n')
    # 2. old core block -> removed (core stays None until after Fj)
    i, j = find_block(d, '        if _need_core:   # opus-engineer a08', '            del R64')
    assert j - i == 4, (i, j)
    del lines[i:j + 1]
    # 3. core after Fj
    k = next(k for k in range(d, len(lines)) if lines[k].startswith('        Fj = _shared_old384_mm('))
    lines[k + 1:k + 1] = NEW_CORE_AFTER_FJ.split('\n')

# the live fallback def = the one captured by the OTT wrapper; the live join = _astra_bcc_push_impl (patch the later one first so
# the earlier line numbers stay valid)
ott = next(k for k in range(len(lines)) if lines[k] == '_OTT_ORIGINAL_PUSH = _shared_old384_push')
owner = max(k for k in range(ott) if lines[k].startswith('def _shared_old384_push('))
impl = [k for k in range(len(lines)) if lines[k].startswith('def _astra_bcc_push_impl(')]
assert len(impl) == 1 and impl[0] > owner
patch_function(impl[0])
patch_function(owner)
text = '\n'.join(lines)
assert text.count('[opus-lead] JSK v1: column-subsampled') == 2 and text.count('[opus-lead] JSK v1: exact Gram core') == 2
# plan constant next to _JOIN_NYSTROM
text = text.replace('\n_JOIN_NYSTROM = True\n', '\n_JOIN_NYSTROM = True\n_JSK_PLAN = %r   # [opus-lead] JSK v1: join layer -> (kc0 start columns, kc power-step columns)\n' % (plan,), 1)
assert '_JSK_PLAN = ' in text
out.write_text(text)
compile(text, str(out), 'exec')
print('wrote', out, 'sha8', hashlib.sha256(text.encode()).hexdigest()[:8], 'plan', plan)
