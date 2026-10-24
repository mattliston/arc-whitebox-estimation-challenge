"""Source-only, reversible E1 pair/GN truncation on frozen Gaussian3 parent."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
H2 = D.parent
PARENT = H2/'hill_climb_precision_g3_symdiag128_v1/candidate.py'
PARENT_SHA = '88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'
EDITS = [
    ('plane_bond_view', 'RR=PB[0:3][:, None]', 'RR=PB[0:2][:, None]'),
    ('plane_product_scratch', 'X2=fnp.reshape(arena[0:45 * nn], (5, 3, 3, n, n))',
     'X2=fnp.reshape(arena[0:30 * nn], (5, 2, 3, n, n))'),
    ('plane_derivative_indices',
     "    _e2_hold.append(fnp.multiply(st['Fr'][[[2, 3, 4], [1, 2, 3], [2, 3, 4], [3, 4, 5], [1, 2, 3]]], st['Gr'][[[1, 2, 3], [2, 3, 4], [2, 3, 4], [1, 2, 3], [3, 4, 5]]], out=X))",
     "    _e2_hold.append(fnp.multiply(st['Fr'][[[2, 3], [1, 2], [2, 3], [3, 4], [1, 2]]], st['Gr'][[[1, 2], [2, 3], [2, 3], [1, 2], [3, 4]]], out=X))"),
    ('gn_unused_rho2',
     "    rho = _g5st['P16'][0]   # G5: = (Cpre / ss) with a zero diagonal, cast: the pair map's rho.astype(float32) / 1\n    rho2 = _e2_H(rho * rho) / 2\n",
     "    rho = _g5st['P16'][0]   # G5: = (Cpre / ss) with a zero diagonal, cast: the pair map's rho.astype(float32) / 1\n    # E1: the GN plane read uses the same retained r=0,1 terms.\n"),
    ('gn_k21_read',
     '    t21a = _e2_H(_e2_H(fnp.outer(A2, A1)) + _e2_H(rho * _e2_H(fnp.outer(A3, A2)))) + _e2_H(rho2 * _e2_H(fnp.outer(A4, A3)))',
     '    t21a = _e2_H(fnp.outer(A2, A1)) + _e2_H(rho * _e2_H(fnp.outer(A3, A2)))'),
    ('gn_k31_read',
     '    t31a = _e2_H(_e2_H(fnp.outer(A3, A1)) + _e2_H(rho * _e2_H(fnp.outer(A4, A2)))) + _e2_H(rho2 * _e2_H(fnp.outer(A5, A3)))',
     '    t31a = _e2_H(fnp.outer(A3, A1)) + _e2_H(rho * _e2_H(fnp.outer(A4, A2)))'),
]


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def save(name, value):
    with (D/name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    assert pin(PARENT.parent/'build.json')['sha256'] == '646bbd8265b6febda17cb204d99cb3407b8e83211274fd902fea17aebee5441b'
    for name in ('candidate.py', 'pricing.json', 'build.json'):
        assert not (D/name).exists(), name
    original = PARENT.read_text(); source = original; edits = []
    for label, before, after in EDITS[:3]:
        assert source.count(before) == 1 and after not in source, label
        source = source.replace(before, after, 1)
        edits.append(dict(label=label, before=before, after=after))
    # The earlier GN definition is a captured fallback. Bind only the final
    # active definition, preserving the fallback and every unrelated byte.
    gn = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_gn_dcov'][-1]
    gn_before = ast.get_source_segment(source, gn); gn_after = gn_before
    for label, before, after in EDITS[3:]:
        assert gn_after.count(before) == 1 and after not in gn_after, label
        gn_after = gn_after.replace(before, after, 1)
    assert source.count(gn_before) == 1 and gn_after not in source
    source = source.replace(gn_before, gn_after, 1)
    edits.append(dict(label='active_gn_matching_plane_reads', before=gn_before, after=gn_after))
    inverse = source
    for edit in reversed(edits):
        assert inverse.count(edit['after']) == 1, edit['label']
        inverse = inverse.replace(edit['after'], edit['before'], 1)
    assert inverse.encode() == PARENT.read_bytes()
    before, after = ast.parse(original), ast.parse(source)
    assert len(before.body) == len(after.body)
    changed = []
    for a, b in zip(before.body, after.body):
        if ast.dump(a) != ast.dump(b):
            assert isinstance(a, ast.FunctionDef) and isinstance(b, ast.FunctionDef) and a.name == b.name
            changed.append(a.name)
    assert changed == ['_e2_pair_state', '_nonlinear_pair_batch', '_e2_pair_arena', '_gn_dcov'], changed
    assert 'Fr16=Fr[1:4], Gr16=Gr[1:4], P16=PB[1:4], P6=PB[1:4][:, None]' in source
    assert 'P12=PB[1:3][:, None]' in source and 'DIV6=fnp.arange(1, 4, dtype=f32)' in source
    assert 'X3=fnp.reshape(arena[0:24 * nn], (4, 2, 3, n, n))' in source
    assert 'T=fnp.reshape(arena[45 * nn:60 * nn], (5, 3, n, n))' in source
    with (D/'candidate.py').open('x') as f: f.write(source)
    nn = 1024**2
    price = dict(source_only=True, width=1024, pair_calls=14, gn_calls=13,
        x2_per_call_saving_flops=45*nn, x2_total_saving_flops=45*nn*14,
        gn_per_call_saving_flops=8*nn, gn_total_saving_flops=8*nn*13,
        dominant_warm_saving_flops=45*nn*14+8*nn*13,
        x2_once_view_bind_saving_flops=15*nn,
        removed_gather_output_elements_per_pair_call=30*1024,
        unpriced=['small advanced-index gather tariff', 'conditional later masks/routes/cache changes'],
        no_new_matmul=True, no_new_allocation=True, parent_arena_bytes_unchanged=True,
        note='Static F32 source-operation delta; full native bills decide the component. Not an RSS/runtime forecast.')
    save('pricing.json', price)
    save('build.json', dict(parent=pin(PARENT), parent_build=pin(PARENT.parent/'build.json'),
        candidate=pin(D/'candidate.py'), builder=pin(__file__), protocol=pin(D/'PROTOCOL.md'),
        primitive=pin(D/'probe.py'), pricing=pin(D/'pricing.json'),
        policy=pin(H2/'hill_climb_policy_20260928_v2.json'), edits=edits,
        source_inverse_exact=True, all_unlisted_source_bytes_unchanged=True,
        changed_function_asts=changed, gaussian_order=3, plane_bond_order=1, local_bond_order=2,
        learned_literals_unchanged=True, source_only=True, numerical_execution=False,
        execution_authorized=False, qualified=False, submitted=False))
    print(json.dumps(pin(D/'candidate.py')))


if __name__ == '__main__': main()
