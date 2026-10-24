"""Stdlib-only append/inverse and active-owner proof; never imports estimator."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
PARENT = D.parent / 'hill_climb_precision_g3_plane_e1_gnbatch_v2/candidate.py'
PARENT_SHA = '06538586dd306df2b00986f43cc85afd39d515a12c8977f4da6fa9c539ed91b1'
FRAGMENT_SHA = '85e75784afb7f3d79854cd5599f4e7f77a15bdef117f6edccb7f23fa85ec70c0'


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def check(parent, child, fragment):
    assert child == parent + '\n' + fragment
    assert child[:-(len(fragment) + 1)] == parent
    pt, ct = ast.parse(parent), ast.parse(child)
    assert [ast.dump(n) for n in ct.body[:len(pt.body)]] == [ast.dump(n) for n in pt.body]
    lines = parent.splitlines()
    def source(n): return '\n'.join(lines[n.lineno - 1:n.end_lineno])
    capture = [n for n in pt.body if isinstance(n, ast.Assign)
               and any(isinstance(x, ast.Name) and x.id == '_OTT_ORIGINAL_PUSH' for x in n.targets)]
    assert len(capture) == 1 and ast.dump(capture[0].value) == "Name(id='_shared_old384_push', ctx=Load())"
    pushes = [n for n in pt.body if isinstance(n, ast.FunctionDef) and n.name == '_shared_old384_push']
    active = [n for n in pushes if n.lineno < capture[0].lineno][-1]
    body = source(active)
    assert body.count('_push_bank_shared_newborn(') == 2  # early no-old fallback plus post-join young
    assert 'wx = w32 if rows is None else w32[:, rows]' in body
    assert "young_newborn = (_e2_H(int(inherited_columns) - old_columns), selected, base, t, s)" in body
    assert 'young = _push_bank_shared_newborn(wx, _e2_H(p[:, old_columns - _cp_old0:]), previous, young_newborn)' in body
    assert body.index("state['columns'] = old_columns") < body.index('young_newborn =')
    assert "answer = fnp.concatenate([old, young], axis=1)" in body
    assert "int(answer.shape[1]) != int(lam.shape[0])" in body
    context = source(pushes[-1])
    for x in ("if 7 <= layer <= 14:", "a=a if rows is None else a[rows]", "live if rows is None else live[rows]",
              "n=int(w32.shape[1]), lam=lam", "_OTT_ORIGINAL_PUSH(layer, w32, p, previous, newborn, lam, state, rows=rows)"):
        assert x in context
    predictor = [n for n in pt.body if isinstance(n, ast.FunctionDef) and n.name == '_predict_binary'
                 and 'newborn = (int(_e2_H(pp.shape[1]))' in source(n)][0]
    pred = source(predictor)
    for x in ("fnp.fill_diagonal(bp, 0)", "_bd_base32[:, selected]", "lam * _S19_LAM[0], _s19_new, -_s19_new",
              "_E2_LAYER['ott_live_mask'] = alive", "diagonal = _ott_diagonal(diagonal, lam)"):
        assert x in pred, x
    assert 'previous, previous_alive = (live_indices, alive)' in pred
    bd = [n for n in pt.body if isinstance(n, ast.FunctionDef) and n.name == '_bd_update'][-1]
    assert 'fnp.fill_diagonal(change, 0.0)' in source(bd)
    assert 'alive_column = alive[:, None]' in source(bd)
    assert 'fnp.multiply(bp + change, alive_column, out=new32)' in source(bd)
    ft = ast.parse(fragment)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom, ast.AugAssign)) for n in ast.walk(ft))
    assert {n.name for n in ft.body if isinstance(n, ast.FunctionDef)} == {
        '_ynt_trace', '_push_bank_shared_newborn', '_ott_diagonal', '_predict_binary'}
    assert hashlib.sha256(fragment.encode()).hexdigest() == FRAGMENT_SHA
    assert '_E1GN' in parent
    return dict(captured_push_line=active.lineno, context_line=pushes[-1].lineno,
                parent_all_function_asts_unchanged=True, wx_context_rows_identical=True,
                postjoin_young_tail_coefficients=True, stored_newborn_zero_diagonal=True,
                original_old_U_core_and_trace_unchanged=True, no_new_setup_kernels=True)


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    assert pin(D / 'fragment.py')['sha256'] == FRAGMENT_SHA
    assert (D / 'fragment.py').read_bytes() == (D.parent / 'hill_climb_young_negative_trace_v1/fragment.py').read_bytes()
    parent = PARENT.read_text(); fragment = (D / 'fragment.py').read_text()
    child = parent + '\n' + fragment; proof = check(parent, child, fragment)
    with (D / 'candidate.py').open('x') as f: f.write(child)
    names = ['candidate_dead_tensor_trace_v1.py', 'dead_tensor_trace_fragment_v1.py',
             'dead_tensor_trace_train2_comparison_v1.json', 'dead_tensor_trace_train2_v1_guard.json',
             'candidate_dead_tensor_conditional_mean_v2.py', 'dead_tensor_conditional_mean_fragment_v2.py',
             'dead_tensor_conditional_mean_train2_comparison_v2.json',
             'hill_climb_precision_g3_plane_e1_gnbatch_v2/build.json',
             'hill_climb_precision_g3_plane_e1_gnbatch_v2/PROTOCOL.md',
             'hill_climb_precision_g3_plane_e1_gnbatch_v2/native8_assessment.json',
             'hill_climb_precision_g3_plane_e1_gnbatch_v2/native8_assess_guard.json',
             'hill_climb_precision_g3_plane_e1_gnbatch_v2/boundary_assessment.json',
             'hill_climb_young_negative_trace_v1/candidate.py',
             'hill_climb_young_negative_trace_v1/build.json',
             'hill_climb_young_negative_trace_v1/PROTOCOL.md',
             'hill_climb_young_negative_trace_v1/native2_assessment.json',
             'hill_climb_young_negative_trace_v1/native2_guard.json',
             'hill_climb_young_negative_trace_v1/native2_assess_guard.json',
             'hill_climb_inherited_young_negative_trace_v1/native2_assessment.json']
    record = dict(parent=pin(PARENT), parent_build=pin(PARENT.parent / 'build.json'),
        candidate=pin(D / 'candidate.py'), builder=pin(__file__), fragment=pin(D / 'fragment.py'),
        primitive=pin(D / 'probe.py'), protocol=pin(D / 'PROTOCOL.md'),
        native_driver=pin(D / 'native_probe.py'),
        policy=pin(D.parent / 'hill_climb_policy_20260928_v2.json'),
        history=[pin(D.parent / n) for n in names], append_only=True,
        suffix_bytes=len(('\n' + fragment).encode()), source_inverse_exact=True, source_proof=proof,
        composition_only=True, unchanged_whole_ynt_fragment_sha256=FRAGMENT_SHA,
        score_comparator=pin(D.parent / 'hill_climb_precision_g3_symdiag128_v1/candidate.py'),
        setup_counts=dict(bound=682, lvfuse=649, psdb_fused=9),
        source_only=True, numerical_execution=False, execution_authorized=False)
    with (D / 'build.json').open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    print(json.dumps(record['candidate']))


if __name__ == '__main__': main()
