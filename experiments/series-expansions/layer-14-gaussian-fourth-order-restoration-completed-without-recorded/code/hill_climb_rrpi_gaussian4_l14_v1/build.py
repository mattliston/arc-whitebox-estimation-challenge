"""Source-only fixed Gaussian Hermite r4 addition at the last full pair layer."""
from pathlib import Path
import ast, hashlib, json

D = Path(__file__).resolve().parent
H = D.parent
P = H / 'hill_climb_pair_rr_inverse_v1/candidate.py'
PARENT_SHA = 'a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
SUFFIX = '''

# One fixed Gaussian rho^4/4! term, only at the last full pair layer.
_G4L14_ENABLED = True
'''
INSERT = '''    if _G4L14_ENABLED and _PHYS_LAYER[0] == 14:
        # P16[2] already holds the original FP32 rho^3/3! recurrence.
        # Leave R3, E1, persistent powers and all subsequent centralization intact.
        _g4_power = _e2_H(P16[2]) * _e2_H(rho / 4.0)
        _g4_fg = _e2_H(FT[4, :, :, None]) * _e2_H(GT[4, :, None, :])
        _g4_term = _e2_H(_g4_fg * _e2_H(_g4_power[None, :, :]))
        _g4_units = _e2_H(fnp.stack((_ps_ss32, _ps_vs32, _ps_vv32)))
        _g4_cov, _g4_pk21, _g4_pk22 = fnp.unstack(_e2_H(_g4_term * _g4_units))
        cov = cov + _g4_cov
        pk21 = pk21 + _g4_pk21
        pk22 = pk22 + _g4_pk22
'''


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def source():
    assert pin(P)['sha256'] == PARENT_SHA
    base = P.read_text()
    tree = ast.parse(base)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_nonlinear_pair_batch']
    assert len(nodes) == 1
    before = ast.get_source_segment(base, nodes[0])
    assert base.count(before) == 1
    anchor = '    m2 = _ps32(2 * m)\n'
    assert before.count(anchor) == 1
    assert "_PHYS_LAYER[0] = l\n            if l in _T31_LAYERS:" in base
    assert 'nonlinear=_nonlinear_pair_batch' in base
    after = before.replace(anchor, INSERT + anchor, 1)
    assert after.replace(INSERT, '', 1) == before
    edits = [dict(before=before, after=after)]
    body = base.replace(before, after, 1)
    assert body.replace(after, before, 1) == base
    candidate = body + SUFFIX
    assert candidate.count('_G4L14_ENABLED = True') == 1
    identity = candidate.replace('_G4L14_ENABLED = True', '_G4L14_ENABLED = False', 1)
    compile(candidate, str(D / 'candidate.py'), 'exec')
    compile(identity, str(D / 'candidate_identity.py'), 'exec')
    return candidate, identity, edits


def main():
    candidate, identity, edits = source()
    prior = pin(H / 'hill_climb_precision_gaussian4_v1/native8_assessment.json')
    assessment = json.loads(Path(prior['path']).read_text())
    assert assessment['passed_integrity'] and not assessment['admitted'] and assessment['n'] == 8
    for name, text in [('candidate.py', candidate), ('candidate_identity.py', identity)]:
        with (D / name).open('x') as out:
            out.write(text)
    record = dict(
        parent=pin(P), candidate=pin(D / 'candidate.py'), identity=pin(D / 'candidate_identity.py'),
        build=pin(__file__), edits=edits, suffix=SUFFIX, whole_source_inverse_exact=True, compiled=True,
        fixed_layer=14, gaussian_order=4, coefficient=1.0,
        layer_binding='_PHYS_LAYER[0] is assigned l immediately before pair_map in the active loop; nonlinear is _nonlinear_pair_batch at L14.',
        formula='For c=0,1,2: raw_pair[c] += (P16[2]*(rho/4))*outer(FT[4,c],GT[4,c])*units[c], units=(ss,vs,vv).',
        channels=['covariance raw channel', 'raw square-linear channel', 'raw square-square channel'],
        precision='Existing FP32 derivative tables, rho, P16 and unit casts; power continues the original divide-then-cumprod recurrence with divisor4.',
        scope='After both packed/unpacked raw pair branches, before unchanged common centralization; subsequent exact marginal diagonal resets remain untouched.',
        unchanged='Existing R3 coefficients, P16/cache contents, E1/GN, ranks, learned parameters and all other source; no coefficient, layer or order sweep.',
        identity_original_array_operations_exact_by_source=True, exact_prefix_layers=list(range(14)),
        prior_gaussian4_assessment=prior,
        prior_scope='Older precision parent: all recurrent pair layers R6->R4; not current RRPI L14-only R3->R4.',
        prior_metrics={key: assessment[key] for key in ('raw_pct', 'bill_pct', 'adjusted_pct', 'adjusted_wins', 'adjusted_ci95_pct')},
        stopped_r3_alpha1_5='Not revived or altered. Existing Gaussian R3 coefficient remains1; this source adds a distinct fixed r4/4! term only at L14.',
        cost='Approximately14*N^2 FP32 arithmetic operations including three separate raw-channel additions and unit multiplications, plus views/stack allocation; no GEMM. Actual native bill/resources required.',
        source_only=True, no_model_or_array_jobs=True)
    with (D / 'source_delta.json').open('x') as out:
        json.dump(record, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({key: record[key] for key in ('candidate', 'identity', 'build')}))


if __name__ == '__main__':
    main()
