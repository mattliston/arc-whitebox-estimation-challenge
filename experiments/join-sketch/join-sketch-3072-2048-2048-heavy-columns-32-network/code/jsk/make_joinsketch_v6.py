"""[opus-lead] JSK-lite (v6): TODAY's Nystrom join sketch (same algorithm, same QR, same noise robustness) but the Gram columns are
computed on the kc HEAVIEST joined columns only (column energy lam_c^2 |J_c|^2): Y = J1 W1 J1[rows]^T (+ the exact old-tier term).
Offline (joinsketch/basis35_v3.txt, basis35_v5_L10/L13.txt): the basis is within +-1.5% of today's at 1/3 of the columns while the
sketch product (2.7% of the bill) shrinks 2-3x. The Nystrom core (the Gram of the subsample in the frame) is rescaled to the full
joined trace (sum (Fj*Fj) @ w) so the next join sees the old tier at full weight (8-11% core error; offline the next basis does not
move). Factors Fj = Qn^T J stay exact over ALL columns. Patches BOTH live join bodies. usage: make_joinsketch_v6.py <parent> <out> [kc json]"""
import json, sys, hashlib
from pathlib import Path
parent, out = Path(sys.argv[1]), Path(sys.argv[2]); assert not out.exists()
kc = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {7: 3072, 10: 2048, 13: 2048}
kc = {int(k): int(v) for k, v in kc.items()}
lines = parent.read_text().split('\n')

def find_block(start_idx, first, last):
    i = next(k for k in range(start_idx, len(lines)) if lines[k].startswith(first))
    j = next(k for k in range(i, len(lines)) if lines[k] == last)
    return i, j

NEW_SKETCH = '''        if _JOIN_NYSTROM:   # [opus-lead] JSK-lite v6: today's Nystrom sketch on the kc heaviest joined columns (jsk/JOINSKETCH_v1.md); the old-tier term and the QR are unchanged
            _jn_JJ = J * J
            _jn_d = fnp.matmul(_jn_JJ, weight)
            _jn_c = fnp.sum(_jn_JJ, axis=0) * weight
            del _jn_JJ
            if old_columns:
                _jn_d = _jn_d + fnp.sum((_shared_old384_mm(Qp, state['core'])) * Qp, axis=1)
            _jn_rows = fnp.sort(fnp.argsort(-(_jn_d.astype(fnp.float64)), kind='stable')[:rank])
            _jn_cols1 = fnp.argsort(-(_jn_c.astype(fnp.float64)), kind='stable')[:_JSK_KC[layer]]
            J1 = J[:, _jn_cols1]
            Y = _shared_old384_mm(J1, weight[_jn_cols1][:, None] * J1[_jn_rows, :].T)
            del J1, _jn_cols1, _jn_c
            if old_columns:
                Y = Y + (_shared_old384_mm(Qp, _shared_old384_mm(state['core'], (Qp[_jn_rows, :]).T)))
            del _jn_d'''

RESCALE_AFTER_FJ = '''        if _need_core:   # [opus-lead] JSK-lite v6: the Nystrom core is the subsample's Gram in the frame; rescale it to the full joined trace
            core = core * (fnp.sum(fnp.matmul(Fj * Fj, weight)) / fnp.trace(core))'''

def patch_function(d):
    i, j = find_block(d, '        if _JOIN_NYSTROM:   # PT-42 Nystrom join', '            del _jn_d')
    assert j - i == 8, (i, j)
    lines[i:j + 1] = NEW_SKETCH.split('\n')
    k = next(k for k in range(d, len(lines)) if lines[k].startswith('        Fj = _shared_old384_mm('))
    lines[k + 1:k + 1] = RESCALE_AFTER_FJ.split('\n')

ott = next(k for k in range(len(lines)) if lines[k] == '_OTT_ORIGINAL_PUSH = _shared_old384_push')
owner = max(k for k in range(ott) if lines[k].startswith('def _shared_old384_push('))
impl = [k for k in range(len(lines)) if lines[k].startswith('def _astra_bcc_push_impl(')]
assert len(impl) == 1 and impl[0] > owner
patch_function(impl[0]); patch_function(owner)
text = '\n'.join(lines)
assert text.count('JSK-lite v6: today') == 2 and text.count('JSK-lite v6: the Nystrom core') == 2
text = text.replace('\n_JOIN_NYSTROM = True\n', '\n_JOIN_NYSTROM = True\n_JSK_KC = %r   # [opus-lead] JSK-lite v6: join layer -> number of heaviest joined columns used for the Nystrom sketch\n' % (kc,), 1)
assert '_JSK_KC = ' in text
out.write_text(text); compile(text, str(out), 'exec')
print('wrote', out, 'sha8', hashlib.sha256(text.encode()).hexdigest()[:8], 'kc', kc)
