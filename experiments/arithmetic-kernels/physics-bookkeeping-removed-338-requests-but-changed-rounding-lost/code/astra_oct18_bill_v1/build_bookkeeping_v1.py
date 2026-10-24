"""Versioned v61j core request reduction, separated exact-reuse/reassociation arms."""
import ast
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent / 'opus_fix336687_v1/candidate_v61j.py'
base = P.read_text()
assert hashlib.sha256(base.encode()).hexdigest() == 'ac93b10f2c8fa541b94cf0787b7f573d36cf1233df219d00a22335c4f9a47129'
tree = ast.parse(base)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_astra_fhm_core')
core = '\n'.join(base.splitlines()[node.lineno-1:node.end_lineno]) + '\n'

def sub(text, before, after, count=1):
    assert text.count(before) == count, (before[:100], text.count(before))
    return text.replace(before, after)

exact = core
exact = sub(exact, '_ps32(2 * bpre)', '(2 * _ps_bpre32)')
exact = sub(exact, 'gq.astype(fnp.float32)', '_ps32(gq)')
term = '(_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (3 * u))) or _e2_value_inline) * u))) or _e2_value_inline)'
exact = sub(exact, term, '_bk_3u2', 2)
exact = sub(exact, '            disc = ', '            _bk_3u2 = (3 * u) * u\n            disc = ')

alg = exact
start = alg.index('                _gn_C0 = C\n')
end = alg.index('                _ps_post_cov32 = C\n', start)
alg = alg[:start] + '''                # Real-arithmetic identity for the same rank-two mean correction.
                # Own all new vectors/matrices. Original C and pair post_v are read-only.
                _rqs2_pair = _E2_LAYER.get('pair')
                _ps_Cdiag0 = (_rqs2_pair['aux']['post_v'] if (_rqs2_pair is not None and _rqs2_pair['cov'] is C) else fnp.diag(C))
                _gn_rc = _GN_RC.get(l, 0.0) - _GN_KC.get(l, 0.0)
                _bk_z64 = m + 0.5 * dm
                _bk_d64 = (1.0 + _gn_rc) * dm if _gn_rc else dm
                _ps_Cdiag = _ps_Cdiag0 - (2 * _bk_z64) * _bk_d64
                _bk_z32 = _ps32(_bk_z64)
                _bk_d32 = _ps32(_bk_d64)
                _e2_mdm = fnp.outer(_bk_z32, _bk_d32)
                _ps_C0 = _ps32(C)
                C = (_ps_C0 - _e2_mdm) - _e2_mdm.T
''' + alg[end:]
alg = sub(alg, '(_plane_cpdiag64 + fnp.diag(star_pre) + _ps_bv * bpre) + _ps_bv * (2 * bpre)', '(_plane_cpdiag64 + fnp.diag(star_pre)) + _ps_bv * (3 * bpre)')
alg = sub(alg, '(radial * (_astra_q64 + 2 * _astra_q64)) / 3', 'radial * _astra_q64')
alg = sub(alg, '_ps_Bdiag * b + _ps_Bdiag * (2 * b)', '_ps_Bdiag * (3 * b)')
alg = sub(alg, '(3 * _ps_rest_diag - 2 * _ps_rest_diag).astype(fnp.float32)', '_ps_rest_diag.astype(fnp.float32)')
alg = sub(alg, "            # Retain the parent's float64 diagonal operation order exactly.", '            # v1 reassociates only exact-real diagonal identities; native endpoint gate required.')

receipt = {'parent': str(P), 'parent_sha256': hashlib.sha256(base.encode()).hexdigest(), 'arms': {}}
for name, body in [('exact', exact), ('algebra', alg)]:
    fragment = '# ASTRA bookkeeping v1: ' + name + '\n' + body + '\n_ASTRA_FHM_ACTIVE_CORE = _tr_clearing(_astra_fhm_core)\n'
    compile(base + '\n' + fragment, '<candidate>', 'exec')
    fragment_path = D / ('fragment_bookkeeping_' + name + '_v1.py')
    candidate = D / ('candidate_v61j_bookkeeping_' + name + '_HELD_v1.py')
    for path, text in [(fragment_path, fragment), (candidate, base + '\n' + fragment)]:
        assert not path.exists(), path
        path.write_text(text)
    receipt['arms'][name] = {'fragment': str(fragment_path), 'candidate': str(candidate), 'sha256': hashlib.sha256(candidate.read_bytes()).hexdigest()}
receipt['static_price'] = {
    'exact_reuse_expected_requests_saved': 60,
    'algebra_plus_exact_expected_requests_saved': 340,
    'explanation': 'Exact per layer: 2 bpre cast1, repeated 3u squared2, cached gq cast1. Algebra additional: mean correction130 total, kdiag30, k22diag45, trace_diag30, star_diag45.',
    'bill': 'Lower arithmetic; approximate mean-covariance saving 4*14*1024^2=58.72MF before vector savings. Must measure whole model.',
    'prediction': 'Algebra preserves real-number physics but changes FP32/FP64 order. No accuracy/score bank claimed.'
}
out = D / 'BOOKKEEPING_BUILD_v1.json'
assert not out.exists()
out.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
