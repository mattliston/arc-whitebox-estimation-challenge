"""Export only an accepted birth-consumer fit and qualify its exact arithmetic."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
V = HERE.parents[2] / 'astra_oct20_vm_frontier_v1'
FIT = RUNS / 'birth_consumer_fit_live_v2/FIT_RESULT_v1.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_SERVER_URL')
    fit = read(FIT)
    assert fit['status'] == 'PASS_TEACHER_FIDELITY_GATE'
    assert fit['source_sha256'] == 'a8926febd6fa8b58010597b7067d6efdb10c22aa4de8a65ea01df6ace6bc2601'
    assert not fit['truth_access'] and not fit['fresh_or_held_targets_used']
    guard = read(RUNS / 'birth_consumer_fit_live_v2_GUARD.json')
    assert guard['successful'] and guard['child_exit'] == 0 and guard['whole_group_retired']
    assert not guard['cleanup_errors'] and guard['max_owned_swap_bytes'] == 0
    parents = {role: V / 'msg8_loaderfix_v1' / ('candidate_msg8_teacher_loaderfix_' + suffix + '_v1.py')
               for role, suffix in [('science', 'sci'), ('release', 'rel')]}
    pins = dict(science='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510',
                release='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05')
    for role, path in parents.items():
        assert sha(path) == pins[role]
    oldA = np.asarray(fit['original_A'], dtype=np.float32)
    newA = np.asarray(fit['fitted_A'], dtype=np.float32)
    assert oldA.shape == newA.shape == (64, 4) and np.isfinite(newA).all()
    assert fit['active_coefficient_columns'] == [0, 2] and fit['dead_columns_unchanged'] == [1, 3]
    np.testing.assert_array_equal(oldA[:, [1, 3]], newA[:, [1, 3]])
    original = parents['science'].read_text()
    tree = ast.parse(original)
    tables = {}
    for name in ('_BDNH_A_LITERAL', '_BD_COEFFICIENT_LITERAL'):
        values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert len(values) == 1
        tables[name] = np.asarray(values[0], dtype=np.float32)
    np.testing.assert_array_equal(oldA, tables['_BDNH_A_LITERAL'])
    out = HERE / 'native_v1'
    out.mkdir(exist_ok=False)
    literal = tuple(tuple(float(x) for x in row) for row in newA)
    tail = '\n# Fixed TRAIN4 birth-consumer teacher readaptation; unchanged runtime operations.\n'
    tail += 'assert _BDNH_PARAMETERS is None and not _E2_BDNH\n'
    tail += '_BDNH_A_LITERAL = ' + repr(literal) + '\n_FAST_RESCUE_FRAC = {}\n'
    tailpath = out / 'tail_birth_consumer_readapt_v1.py'
    with tailpath.open('x') as f:
        f.write(tail)
    outputs = {}
    for role, parent in parents.items():
        source = parent.read_text() + tail
        path = out / ('candidate_birth_consumer_readapt_' + role + '_v1.py')
        compile(source, str(path), 'exec')
        old = {n.name: ast.dump(n) for n in ast.parse(parent.read_text()).body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        new = {n.name: ast.dump(n) for n in ast.parse(source).body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert old == new
        with path.open('x') as f:
            f.write(source)
        outputs[role] = dict(path=str(path), sha256=sha(path), parent_path=str(parent), parent_sha256=sha(parent))
    write(out / 'BUILD_v1.json', dict(utc=utc(), outputs=outputs, tail_sha256=sha(tailpath),
          fit_sha256=sha(FIT), only_changed_parameter='_BDNH_A_LITERAL', runtime_shapes_unchanged=True,
          all_functions_and_classes_AST_unchanged=True, truth_access=False, no_native_launch=True))
    # Execute the exact original birth function through the official SDK.
    environment = dict(fnp=fnp, _E2_BDNH={})
    for name in ('_E2Hold', '_e2_bdnh_coefficients'):
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
        assert len(nodes) == 1
        node = ast.Module(body=nodes, type_ignores=[])
        exec(compile(node, str(parents['science']) + ':birth_tiny', 'exec'), environment)
    fn = environment['_e2_bdnh_coefficients']
    rng = np.random.default_rng(20261021)
    cases = []
    with fs.BudgetContext(flop_budget=2**34, quiet=True) as context:
        for n in (16, 1024):
            H = rng.normal(size=(n, 64)).astype(np.float32)
            nativeH = fnp.asarray(H)
            cstar = fnp.asarray(tables['_BD_COEFFICIENT_LITERAL'])
            nativeA = [fnp.asarray(x) for x in (oldA, newA)]
            for layer in range(1, 15):
                products = []
                for A, handle in zip((oldA, newA), nativeA):
                    environment['_E2_BDNH'].clear()
                    runs = []
                    for repetition in range(2):
                        begin, calls = context.flops_used, len(context._op_log)
                        value = fn(nativeH, handle, cstar, layer)
                        cost = (context.flops_used - begin, len(context._op_log) - calls)
                        array = np.asarray(value).copy()
                        runs.append((array, cost))
                    np.testing.assert_array_equal(runs[0][0], runs[1][0])
                    centered = H - H.mean(axis=0)
                    Z = np.clip(centered / np.sqrt((centered * centered).mean(axis=0) + np.float32(1e-6)), -8, 8)
                    c = tables['_BD_COEFFICIENT_LITERAL'][layer]
                    oracle = (c + (np.float32(1) - np.abs(c)) * np.tanh(Z @ A)).T
                    np.testing.assert_allclose(runs[1][0], oracle, rtol=3e-5, atol=2e-6)
                    assert np.isfinite(runs[1][0]).all()
                    products.append(runs)
                assert products[0][0][1] == products[1][0][1]
                assert products[0][1][1] == products[1][1][1]
                np.testing.assert_array_equal(np.asarray(nativeH), H)
                cases.append(dict(n=n, layer=layer, cold_tariff=products[0][0][1], warm_tariff=products[0][1][1],
                                  repeated_bytes_exact=True, input_ownership_preserved=True, tariff_unchanged=True))
    write(out / 'TINY_OFFICIAL_v1.json', dict(utc=utc(), successful=True, official_version=fs.__version__,
          cases=cases, source_sha256=outputs['science']['sha256'], physical_oracle=True, no_model_predictions=True))
    subprocess.run(['/home/arcworker/arc-envs/client/bin/python', '-B', str(HERE / 'check_import_v2.py'),
                    '--build', str(out / 'BUILD_v1.json'), '--out', str(out / 'NUMPY_FREE_IMPORT_v1.json')], check=True)
    planpath = V / 'PLAN_BANK8_ACTUAL_SHARED_v1.json'
    assert sha(planpath) == 'cf76262261a85fcb6b5c508ad37a2b13f92884b6b69aff66bd0da4bcd0833793'
    plan = read(planpath)
    science = Path(outputs['science']['path'])
    plan.update(utc=utc(), kind='CURRENT_MSG8_BIRTH_CONSUMER_READAPTATION', reference_submission=337422,
                fit_sha256=sha(FIT), candidates={'birth_consumer_readapt': dict(path=str(science.relative_to(ROOT)),
                sha256=sha(science), effective_sha256=sha(science), bytes=science.stat().st_size)})
    target = HERE / 'PLAN_BIRTH_CONSUMER_READAPT_v1.json'
    write(target, plan)
    ready = dict(utc=utc(), successful=True, candidate='birth_consumer_readapt', outputs=outputs,
                 plan=str(target), plan_sha256=sha(target), fit_sha256=sha(FIT),
                 tiny_sha256=sha(out / 'TINY_OFFICIAL_v1.json'), import_sha256=sha(out / 'NUMPY_FREE_IMPORT_v1.json'),
                 source_sha256=outputs['science']['sha256'], jobs=12, shared_max_active=36, blas_threads=4,
                 native_launches=0, adjusted_score_gain=None, reference_submission=337422)
    write(out / 'READY_v1.json', ready)
    print(json.dumps(ready))


if __name__ == '__main__':
    main()
