"""Stdlib-only reversible GN channel batching on the preserved failed E1 source."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
H2 = D.parent
PARENT = H2/'hill_climb_precision_g3_plane_e1_v1/candidate.py'
PARENT_SHA = '6ac194efda6491b48f86eca074493c503f9312c6fd9e667105f378ee7637d6f9'


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    assert not (D/'candidate.py').exists() and not (D/'build.json').exists()
    original = PARENT.read_text(); nodes = ast.parse(original).body
    node = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == '_gn_dcov'][-1]
    before = ast.get_source_segment(original, node); after = before
    old_views = "    A1, A2, A3, A4, A5 = fnp.unstack(_e2_H(_g5st['FT'][1:6, 0]))   # G5: the pair map stacked aa.astype(float32) = derivatives(a, pdf, g, mg/s, 6) rows into FT[:, 0]"
    old_div = "    dT21 = dK21 / _e2_H(_g5_outer(_ps32(v), _ps32(s)))"
    old_d31 = next(line for line in before.splitlines() if line.startswith('    dT31 = _g5_rowmul('))
    weight = old_d31[len('    dT31 = _g5_rowmul('):].split(', rho)   # G5:', 1)[0]
    assert weight.endswith('))')
    old_contract = "    t21a = _e2_H(fnp.outer(A2, A1)) + _e2_H(rho * _e2_H(fnp.outer(A3, A2)))\n    t31a = _e2_H(fnp.outer(A3, A1)) + _e2_H(rho * _e2_H(fnp.outer(A4, A2)))\n    X = _e2_H(_e2_H(dT21 * t21a) / 2) + _e2_H(_e2_H(dT31 * t31a) / 6)"
    changes = [
        (old_views, "    _e1gn = _e1gn_state(_g5st, int(mu.shape[0]))"),
        (old_div, "    _e2_H(fnp.divide(dK21, _e2_H(_g5_outer(_ps32(v), _ps32(s))), out=_e1gn['D0']))"),
        (old_d31, "    _e2_H(fnp.multiply("+weight+", _e1gn['rhoT'], out=_e1gn['D1T']))   # same weight_i * rho_ij, no einsum out-copy"),
        (old_contract, "    X = _e1gn_contract(_e1gn, rho, _e2_H)"),
    ]
    for a, b in changes:
        assert after.count(a) == 1 and b not in after
        after = after.replace(a, b, 1)
    assert original.count(before) == 1
    prefix = original.replace(before, after, 1)
    suffix = '\n'+(D/'fragment.py').read_text()
    source = prefix+suffix
    assert source[:-len(suffix)].replace(after, before, 1) == original
    a, b = ast.parse(original), ast.parse(prefix)
    assert len(a.body) == len(b.body)
    changed = [(x.name, x.lineno) for x,y in zip(a.body,b.body) if ast.dump(x)!=ast.dump(y)]
    assert changed == [('_gn_dcov', node.lineno)]
    ast.parse(source)
    # Byte-bound final arithmetic and fallback remain exactly the parent's.
    assert after.split('    dcov = ', 1)[1] == before.split('    dcov = ', 1)[1]
    assert after.split('    _g5st = ', 1)[0] == before.split('    _g5st = ', 1)[0]
    with (D/'candidate.py').open('x') as f: f.write(source)
    result = dict(parent=pin(PARENT), parent_build=pin(PARENT.parent/'build.json'),
        comparator=pin(H2/'hill_climb_precision_g3_symdiag128_v1/candidate.py'),
        candidate=pin(D/'candidate.py'), builder=pin(__file__), fragment=pin(D/'fragment.py'),
        protocol=pin(D/'PROTOCOL.md'), primitive=pin(D/'probe.py'),
        policy=pin(H2/'hill_climb_policy_20260928_v2.json'),
        edits=[dict(label='active_gn_channel_batch',before=before,after=after)],
        suffix_bytes=len(suffix.encode()), source_inverse_exact=True,
        all_unlisted_source_bytes_unchanged=True, changed_function_asts=changed,
        gn_calls=13, warm_arithmetic_calls_before=13, warm_arithmetic_calls_after=7,
        warm_view_calls_removed=2, projected_explicit_calls_saved=104,
        warm_arithmetic_flops_unchanged=True, new_arena_allocation=False,
        borrowed_scratch_elements=6*1024**2, once_reshape_flops=6*1024**2,
        source_only=True, numerical_execution=False, execution_authorized=False)
    with (D/'build.json').open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result['candidate']))


if __name__ == '__main__': main()
