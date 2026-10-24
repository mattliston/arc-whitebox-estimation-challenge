"""Root-only paid n=4 retained-term fixture; no estimator import.
NATIVE -B probe.py BUILD_SHA RESULT, guard60s/1GB/zero added swap, BLAS1.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D = Path(__file__).resolve().parent

def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

def extract(source, n):
    nodes = ast.parse(source).body
    definitions = {node.name: node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    env = dict(fnp=fnp, _E2_PAIR_STATE={}, _E2_PAIR_ARENA={}, _E1GN_CACHE={},
               _PHYS_LAYER=[None], _SCRATCH=[fnp.empty(72*n*n, dtype=fnp.float32)])
    names = ['_E2Hold', '_e2_pair_state', '_e2_pair_arena', '_e1gn_state', '_e1gn_contract']
    exec(compile(ast.Module(body=[definitions[name] for name in names], type_ignores=[]),
                 '<pinned definitions>', 'exec'), env)
    body = definitions['_nonlinear_pair_batch'].body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and (node.targets[0].id == '_late_plane_e0'
                      or (node.targets[0].id == 'X' and ast.unparse(node.value) == "_e2_A['X2']")))
    end = next(i for i, node in enumerate(body[start:], start) if isinstance(node, ast.Assign)
               and isinstance(node.targets[0], ast.Tuple)
               and [item.id for item in node.targets[0].elts] == ['T1', 'T2', 'T3', 'T4', 'T5'])
    code = 'def actual_pair(st, _e2_A):\n    _e2_H = _E2Hold()\n    _e2_hold = []\n'
    code += '\n'.join('    '+line for node in body[start:end+1] for line in ast.unparse(node).splitlines())
    code += '\n    return T1, T2, T3, T4, T5\n'
    exec(compile(code, '<pinned pair statements>', 'exec'), env)
    return env

def reference(F, G, rho, order):
    result = np.zeros((5, 3, len(rho), len(rho)), dtype=np.float64)
    for channel, (a, b) in enumerate(((2, 1), (1, 2), (2, 2), (3, 1), (1, 3))):
        for r in range(order+1):
            weight = 1 if r == 0 else rho.astype(np.float64)
            result[channel] += weight * F[a+r, :, :, None].astype(np.float64) * G[b+r, :, None, :].astype(np.float64)
    return result

def same_bytes(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def paid(ctx, function, *args):
    before = int(ctx.summary_dict()['flops_used'])
    value = function(*args)
    return value, int(ctx.summary_dict()['flops_used']) - before

def main():
    assert len(sys.argv) == 3
    target = Path(sys.argv[2]).resolve()
    assert not target.exists()
    assert Path(sys.executable).absolute() == D.parent.parent / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
    assert np.__version__ == '2.4.6'
    assert all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
    build_pin = pin(D / 'build.json')
    assert build_pin['sha256'] == sys.argv[1]
    build = json.loads((D / 'build.json').read_text())
    for key in ('parent', 'candidate', 'builder', 'protocol', 'primitive'):
        assert build[key] == pin(build[key]['path'])
    assert build['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    parent = Path(build['parent']['path']).read_text()
    source = Path(build['candidate']['path']).read_text()
    inverse = source
    for edit in reversed(build['edits']):
        assert inverse.count(edit['after']) == 1
        inverse = inverse.replace(edit['after'], edit['before'], 1)
    assert inverse == parent and build['source_inverse_exact']
    n = 4
    F = ((np.arange(7*3*n).reshape(7, 3, n) % 7)-3).astype(np.float32)
    G = ((np.arange(7*3*n).reshape(7, 3, n)*3 % 5)-2).astype(np.float32)
    rho = ((np.arange(n*n).reshape(n, n) % 5)-2).astype(np.float32)/4
    rho = (rho+rho.T)/2
    np.fill_diagonal(rho, 0)
    zero = np.zeros_like(rho)
    zero.flat[::2] = np.float32(-0.0)
    cases = []
    for case, initial_rho in (('zero_signed', zero), ('signed', rho), ('opposite', -rho)):
        results = {}
        for arm, text in (('parent', parent), ('candidate', source)):
            with flops.BudgetContext(flop_budget=10_000_000, quiet=True) as ctx:
                env = extract(text, n)
                st = env['_e2_pair_state'](n)
                scratch = env['_e2_pair_arena'](n)
                gn = env['_e1gn_state'](st, n)
                for layer in (9, 10, 14, 15):
                    for turn in (0, 1):
                        current_F = F + np.float32(turn/2)
                        current_G = G - np.float32(turn/4)
                        current_G[:, 0] = current_F[:, 0]
                        current_rho = initial_rho if turn == 0 else -initial_rho
                        powers = np.stack((np.ones_like(current_rho), current_rho))[:, None]
                        env['_PHYS_LAYER'][0] = layer
                        fnp.copyto(st['FT'], fnp.asarray(current_F))
                        fnp.copyto(st['GT'], fnp.asarray(current_G))
                        fnp.copyto(st['RR'], fnp.asarray(powers))
                        output, pair_flops = paid(ctx, env['actual_pair'], st, scratch)
                        output = np.stack([np.asarray(value).copy() for value in output])
                        order = 0 if arm == 'candidate' and 10 <= layer <= 14 else 1
                        expected = reference(current_F, current_G, current_rho, order)
                        assert output.dtype == np.float32 and np.array_equal(output.astype(np.float64), expected)
                        d21 = rho + np.float32(turn/4)
                        d31 = rho.T/2 - np.float32(turn/8)
                        factors = np.stack((d21, d31))
                        fnp.copyto(gn['D'], fnp.asarray(factors))
                        actual_gn, gn_flops = paid(ctx, env['_e1gn_contract'], gn,
                                                  st['P16'][0], env['_E2Hold']())
                        actual_gn = np.asarray(actual_gn).copy()
                        # Independent FP32 GN operation order also preserves signed zero.
                        terms = np.stack((current_F[2, 0, :, None]*current_F[1, 0, None, :],
                                          current_F[3, 0, :, None]*current_F[1, 0, None, :]))
                        if order == 1:
                            tail = np.stack((current_F[3, 0, :, None]*current_F[2, 0, None, :],
                                             current_F[4, 0, :, None]*current_F[2, 0, None, :]))
                            terms = terms + current_rho*tail
                        assert np.array_equal(terms, np.stack((expected[0, 0], expected[3, 0])).astype(np.float32))
                        channels = factors * terms
                        channels = channels / np.array([2, 6], dtype=np.float32)[:, None, None]
                        expected_gn = channels[0] + channels[1]
                        same_bytes(actual_gn, expected_gn)
                        assert gn_flops == (7 if order == 0 else 13) * n*n
                        same_bytes(np.asarray(st['FT']), current_F)
                        same_bytes(np.asarray(st['GT']), current_G)
                        same_bytes(np.asarray(st['RR']), powers)
                        assert scratch['X1'].shape == (3, 3, n, n)
                        assert scratch['X3'].shape == (4, 2, 3, n, n)
                        assert st['P16'].shape == (3, n, n) and st['P12'].shape == (2, 1, n, n)
                        assert np.isfinite(output).all() and np.isfinite(actual_gn).all()
                        results[arm, layer, turn] = dict(pair=output, gn=actual_gn,
                                                        pair_flops=pair_flops, gn_flops=gn_flops)
        rows = []
        for layer in (9, 10, 14, 15):
            for turn in (0, 1):
                old, new = results['parent', layer, turn], results['candidate', layer, turn]
                if layer in (9, 15):
                    same_bytes(old['pair'], new['pair'])
                    same_bytes(old['gn'], new['gn'])
                    assert old['pair_flops'] == new['pair_flops'] and old['gn_flops'] == new['gn_flops']
                else:
                    assert old['pair_flops'] > new['pair_flops']
                    assert old['gn_flops'] - new['gn_flops'] == 6*n*n
                rows.append(dict(layer=layer, turn=turn, retained_terms_exact=True,
                    matched_gn_exact=True, inputs_immutable=True,
                    parent_pair_flops=old['pair_flops'], child_pair_flops=new['pair_flops'],
                    parent_gn_flops=old['gn_flops'], child_gn_flops=new['gn_flops']))
        cases.append(dict(case=case, rows=rows, unaffected_layers_byte_exact=True))
    result = dict(passed=True, candidate=build['candidate'], parent=build['parent'],
        build=build_pin, probe=pin(__file__), cases=cases, source_inverse_exact=True,
        retained_r0_all_five_three_channel_outputs_exact=True, matched_gn_exact=True,
        unchanged_gaussian_local_shapes=True, repeated_state_fresh=True,
        model_executed=False, release_admitted=False)
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
