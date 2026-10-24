"""[opus-lead] JSK v4 = whitened column-subsampled power-step sketch (v2) + a FULL-TRACE core. Build <out> from <parent>, patching BOTH
live join bodies (_astra_bcc_push_impl and the OTT-captured _shared_old384_push fallback):
  * start Y1 = G0[:, rows] on the kc0 heaviest joined columns; old tier folded as 384 exact columns Qp L_c (core = L_c L_c^T) when the
    core's f32 Cholesky succeeds, else kept as the Qp core Qp^T term (v2 form);
  * whitening Y1 <- Y1 L^-T (L L^T = Y1[rows]; one cholesky + one solve), QR fallback;
  * one power step Y = G1 Y1w on the kc heaviest columns (+ folded old columns);
  * core for the next join = exact Gram of the kc heaviest joined columns in the new frame, RESCALED to the full joined trace, plus the
    inherited old-tier Gram overlap core_old overlap^T (today's Nystrom core is exact to 0.1%; v2's core carried 54-65% of the trace and
    degraded L13/L14 on 6/7 ranks).
usage: make_joinsketch_v4.py <parent.py> <out.py> [plan json]"""
import json, sys, hashlib
from pathlib import Path
parent, out = Path(sys.argv[1]), Path(sys.argv[2]); assert not out.exists()
plan = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {7: (1024, 2048), 10: (1024, 1536), 13: (768, 1536)}
plan = {int(k): tuple(int(x) for x in v) for k, v in plan.items()}
lines = parent.read_text().split('\n')

def find_block(start_idx, first, last):
    i = next(k for k in range(start_idx, len(lines)) if lines[k].startswith(first))
    j = next(k for k in range(i, len(lines)) if lines[k] == last)
    return i, j

NEW_SKETCH = '''        if _JOIN_NYSTROM:   # [opus-lead] JSK v4 (jsk/JOINSKETCH_v1.md): column-subsampled Gram, whitened start, one power step; old tier folded as columns Qp L_c when possible
            _jn_JJ = J * J
            _jn_d = fnp.matmul(_jn_JJ, weight)
            _jn_c = fnp.sum(_jn_JJ, axis=0) * weight
            del _jn_JJ
            _jn_kc0, _jn_kc = _JSK_PLAN[layer]
            _jn_order = fnp.argsort(-(_jn_c.astype(fnp.float64)), kind='stable')
            _jn_cols1 = _jn_order[:_jn_kc]
            J1 = J[:, _jn_cols1]
            J0 = J[:, _jn_order[:_jn_kc0]]
            _jn_w1 = weight[_jn_cols1]
            _jn_w0 = weight[_jn_order[:_jn_kc0]]
            del _jn_c, _jn_order
            _jn_QL = None
            if old_columns:
                try:
                    _jn_Lc = fnp.linalg.cholesky(state['core'] + fnp.eye(int(state['core'].shape[0]), dtype=fnp.float32) * (fnp.float32(1e-5 / int(state['core'].shape[0])) * fnp.trace(state['core'])))
                    _jn_QL = _shared_old384_mm(Qp, _jn_Lc)
                    del _jn_Lc
                    _jn_d = _jn_d + fnp.sum(_jn_QL * _jn_QL, axis=1)
                    J1 = fnp.concatenate([J1, _jn_QL], axis=1)
                    J0 = fnp.concatenate([J0, _jn_QL], axis=1)
                    _jn_ones = fnp.ones(int(_jn_QL.shape[1]), dtype=weight.dtype)
                    _jn_w1 = fnp.concatenate([_jn_w1, _jn_ones])
                    _jn_w0 = fnp.concatenate([_jn_w0, _jn_ones])
                    del _jn_ones
                except Exception:
                    _jn_QL = None
                    _jn_Qc = _shared_old384_mm(Qp, state['core'])
                    _jn_d = _jn_d + fnp.sum(_jn_Qc * Qp, axis=1)
            _jn_rows = fnp.sort(fnp.argsort(-(_jn_d.astype(fnp.float64)), kind='stable')[:rank])
            Y = _shared_old384_mm(J0, _jn_w0[:, None] * J0[_jn_rows, :].T)
            if old_columns and _jn_QL is None:
                Y = Y + _shared_old384_mm(_jn_Qc, Qp[_jn_rows, :].T)
            del J0, _jn_w0, _jn_d
            # whiten the start (float32: the power step squares the spectrum and would drown the trailing directions)
            _jn_K1 = Y[_jn_rows, :]
            _jn_K1 = _jn_K1 + fnp.eye(rank, dtype=fnp.float32) * (fnp.float32(1e-5 / rank) * fnp.trace(_jn_K1))
            try:
                _jn_Yw = fnp.linalg.solve(fnp.linalg.cholesky(_jn_K1), Y.T).T
            except Exception:
                _jn_Yw, _jn_R1 = fnp.linalg.qr(Y, mode='reduced')
                del _jn_R1
            del _jn_K1, Y
            Y = _shared_old384_mm(J1, _jn_w1[:, None] * _shared_old384_mm(J1.T, _jn_Yw))
            if old_columns and _jn_QL is None:
                Y = Y + _shared_old384_mm(_jn_Qc, _shared_old384_mm(Qp.T, _jn_Yw))
                del _jn_Qc
            del J1, _jn_w1, _jn_Yw, _jn_QL'''

NEW_CORE_AFTER_FJ = '''        if _need_core:   # [opus-lead] JSK v4: exact Gram core of the kc heaviest joined columns in the new frame, rescaled to the full joined trace (the inherited old-tier Gram is added below)
            _jn_F1 = Fj[:, _jn_cols1]
            core = _shared_old384_mm(_jn_F1 * weight[_jn_cols1][None, :], _jn_F1.T)
            core = core * (fnp.sum(fnp.matmul(Fj * Fj, weight)) / fnp.trace(core))
            del _jn_F1
        del _jn_cols1'''

INHERITED = '''            if _need_core:   # [opus-lead] JSK v4: inherited old-tier Gram in the new frame
                core = _shared_old384_mm(overlap, _shared_old384_mm(state['core'], overlap.T)) + core
                core = ((core + core.T) * 0.5).astype(fnp.float32)'''

def patch_function(d):
    i, j = find_block(d, '        if _JOIN_NYSTROM:   # PT-42 Nystrom join', '            del _jn_d')
    assert j - i == 8, (i, j)
    lines[i:j + 1] = NEW_SKETCH.split('\n')
    i, j = find_block(d, '        if _need_core:   # opus-engineer a08', '            del R64')
    assert j - i == 4, (i, j)
    del lines[i:j + 1]
    k = next(k for k in range(d, len(lines)) if lines[k].startswith('        Fj = _shared_old384_mm('))
    lines[k + 1:k + 1] = NEW_CORE_AFTER_FJ.split('\n')
    k = next(k for k in range(d, len(lines)) if lines[k].startswith('            overlap = _shared_old384_mm('))
    assert lines[k + 1].startswith('            factors = _slr_join_factors(layer, overlap, state, Fj)'), lines[k + 1]
    lines[k + 1:k + 1] = INHERITED.split('\n')

ott = next(k for k in range(len(lines)) if lines[k] == '_OTT_ORIGINAL_PUSH = _shared_old384_push')
owner = max(k for k in range(ott) if lines[k].startswith('def _shared_old384_push('))
impl = [k for k in range(len(lines)) if lines[k].startswith('def _astra_bcc_push_impl(')]
assert len(impl) == 1 and impl[0] > owner
patch_function(impl[0]); patch_function(owner)
text = '\n'.join(lines)
assert text.count('[opus-lead] JSK v4 (jsk/') == 2 and text.count('JSK v4: exact Gram core') == 2 and text.count('JSK v4: inherited') == 2
text = text.replace('\n_JOIN_NYSTROM = True\n', '\n_JOIN_NYSTROM = True\n_JSK_PLAN = %r   # [opus-lead] JSK v4: join layer -> (kc0 start columns, kc power-step columns)\n' % (plan,), 1)
assert '_JSK_PLAN = ' in text
out.write_text(text); compile(text, str(out), 'exec')
print('wrote', out, 'sha8', hashlib.sha256(text.encode()).hexdigest()[:8], 'plan', plan)
