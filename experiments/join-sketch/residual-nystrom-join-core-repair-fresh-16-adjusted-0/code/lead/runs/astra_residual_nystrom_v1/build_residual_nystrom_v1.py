"""One exact source edit in the active matched-C join, no estimator calls."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
O = D.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')


parent = O/'scout_native/k4liab_s51k2tw/candidate.py'
assert sha(parent) == '6a145fadb3224501e683e95424bd9ada0ff571041850c7d482969b742d8f8f41'
source = parent.read_text()
marker = '# Approximate range-finder core, original orthogonal factor projection.\ndef _shared_old384_push('
assert source.count(marker) == 1
start = source.index(marker)
stop = source.index('\ndef ', start + len(marker))
body = source[start:stop]
old = """            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])
            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_e2_H(Qp[_jn_rows, :]).T)))))
"""
assert body.count(old) == 1
new = """            if layer == 10:
                if rank != 320:
                    raise RuntimeError('RNS requires unchanged L10 rank320')
                Y, _jn_rows = _rns_join_columns(J, weight, Qp if old_columns else None,
                                               state['core'] if old_columns else None, _jn_d)
            else:
""" + ''.join('    ' + line + '\n' for line in old.splitlines())
changed = body.replace(old, new)
fragment = D/'residual_nystrom_fragment_v1.py'
candidate_source = source[:start] + changed + source[stop:] + '\n' + fragment.read_text()
ast.parse(candidate_source)
candidate = D/'matched_c_residual_nystrom_v1/candidate.py'
write(candidate, candidate_source)
assert (source[:start] + changed.replace(new, old) + source[stop:]) == source
write(D/'SOURCE_PROOF_v1.json', dict(
    status='SOURCE_READY_HELD_NO_MODELS', parent_path=str(parent), parent_sha256=sha(parent),
    candidate_path=str(candidate), candidate_sha256=sha(candidate), fragment_sha256=sha(fragment),
    active_core_single_block_edit=True, reverse_full_source_exact=True,
    captured_OTT_core_preserved=True, unchanged_QR_rank_core_solve_factors_consumers=True,
    fixed_layer=10, fixed_split=[192,128], fixed_rank=320, regularizer_relative=1e-6,
    conditioning='symmetrized192x192 block plus fixed ridge; Cholesky must pass; no retry, fallback or ridge escalation',
    selected_basis_is_new_science=True, no_accuracy_or_price_claim=True))
print(json.dumps(dict(candidate_sha256=sha(candidate), fragment_sha256=sha(fragment))))
