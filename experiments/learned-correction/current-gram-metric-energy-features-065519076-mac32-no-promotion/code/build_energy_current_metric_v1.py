"""One fixed full-current-Gram E2/E3 seed, preserving E1 and all messages."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zlib
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/arcworker/arc-workspace')
V = HERE.parents[2] / 'astra_oct20_vm_frontier_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def physical(T, X, root=None):
    mu = np.mean(X, axis=1)
    centered = X - mu[:, None, :]
    wide = centered.astype(np.float64)
    gram = wide.transpose(0, 2, 1) @ wide
    if root is not None:
        gram = (root @ gram) @ root
    _, vectors = np.linalg.eigh(gram)
    V2 = vectors[:, :, -2:] if root is None else root @ vectors[:, :, -2:]
    coefficients = centered @ V2.astype(np.float32)
    projected = T @ coefficients
    Q, _ = np.linalg.qr(projected.astype(np.float64), mode='reduced')
    Q = Q.astype(np.float32)
    back = Q.transpose(0, 2, 1) @ T
    fitted = back @ centered
    return Q @ fitted + np.sum(T, axis=2)[:, :, None] * mu[:, None, :]


def main():
    assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_SERVER_URL')
    parents = {role: V / 'msg8_loaderfix_v1' / ('candidate_msg8_teacher_loaderfix_' + suffix + '_v1.py')
               for role, suffix in [('science', 'sci'), ('release', 'rel')]}
    pins = dict(science='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510',
                release='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05')
    for role, p in parents.items(): assert sha(p) == pins[role]
    source = parents['science'].read_text(); tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_GNN_LITERAL' for t in n.targets)]
    literal = ast.literal_eval(nodes[-1].value)
    shape, packed = literal['enc0_w']; assert tuple(shape) == (128, 78)
    encoder = np.frombuffer(zlib.decompress(base64.b64decode(packed)), dtype='<f4').reshape(shape)
    metrics = np.zeros((3, 128, 8), dtype=np.float64)
    metrics[0, :8] = np.eye(8)
    metrics[1] = encoder[:, 62:70]
    metrics[2] = encoder[:, 70:78]
    grams = metrics.transpose(0, 2, 1) @ metrics
    values, vectors = np.linalg.eigh(grams)
    assert np.isfinite(values).all() and values.min() > 1e-12
    roots = (vectors * np.sqrt(values)[:, None, :]) @ vectors.transpose(0, 2, 1)
    np.testing.assert_array_equal(roots[0], np.eye(8))
    old_gnn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_gop2_gnn_step')
    gnn = ast.get_source_segment(source, old_gnn)
    assert gnn.count('E = _gop2_contract(') == 1
    changed_gnn = gnn.replace('E = _gop2_contract(', 'E = _ecm8_energy_contract(')
    expected_ast = ast.parse(gnn).body[0]
    changed_calls = 0
    for n in ast.walk(expected_ast):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_gop2_contract':
            n.func.id = '_ecm8_energy_contract'; changed_calls += 1
    assert changed_calls == 1 and ast.dump(expected_ast) == ast.dump(ast.parse(changed_gnn).body[0])
    fragment = (HERE / 'fragment_energy_current_metric_v1.py').read_text()
    tail = '\n' + fragment + '\n_ECM8_ENCODER_LITERAL = ' + repr(metrics.tolist())
    tail += '\n\n' + changed_gnn + '\n_FAST_RESCUE_FRAC = {}\n'
    out = HERE / 'native_v1'; out.mkdir(exist_ok=False)
    tailpath = out / 'tail_energy_current_metric_v1.py'; tailpath.write_text(tail)
    outputs = {}
    for role, parent in parents.items():
        path = out / ('candidate_energy23_current_metric_' + role + '_v1.py')
        result = parent.read_text() + tail
        compile(result, str(path), 'exec')
        with path.open('x') as f: f.write(result)
        outputs[role] = dict(path=str(path), sha256=sha(path), parent_path=str(parent), parent_sha256=sha(parent))
    write(out / 'BUILD_v1.json', dict(utc=utc(), outputs=outputs, tail_sha256=sha(tailpath),
        prereg_sha256=sha(HERE / 'PREREG_FIXED_E23_v1.json'), fitting=False, truth_access=False,
        energy_E1_metric='identity', energy_E2_E3_current_enc0_columns=[[62, 70], [70, 78]],
        one_GNN_call_replacement=True, all_other_parent_code_unchanged=True,
        exact_full_mean=True, rank=2, M2_W3_unchanged=True,
        metric_eigenvalues=values.tolist(), native_launches=0))
    env = dict(fnp=fnp, _ECM8_ROOT=None, _ECM8_ENCODER_LITERAL=metrics.tolist())
    old_contract = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_gop2_contract')
    new_contract = next(n for n in ast.parse(fragment).body if isinstance(n, ast.FunctionDef))
    for node in (old_contract, new_contract):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(parents['science']) + ':energy_tiny', 'exec'), env)
    rng = np.random.default_rng(20261021); cases = []
    with fs.BudgetContext(flop_budget=2**34, quiet=True) as context:
        for n in (16, 128, 1024):
            X = np.square(rng.normal(size=(3, n, 8))).astype(np.float32)
            w = rng.normal(scale=n**-.5, size=(n, n)).astype(np.float32)
            T = np.stack([w**2, w**4, w**6])
            nx, nt = fnp.asarray(X), fnp.asarray(T)
            before, calls = context.flops_used, len(context._op_log)
            parent = env['_gop2_contract'](nt, nx)
            parent_cost = [context.flops_used-before, len(context._op_log)-calls]
            runs = []
            env.update(_ECM8_ROOT=None, _ECM8_ENCODER_LITERAL=metrics.tolist())
            for repeat in range(2):
                before, calls = context.flops_used, len(context._op_log)
                result = env['_ecm8_energy_contract'](nt, nx)
                cost = [context.flops_used-before, len(context._op_log)-calls]
                runs.append((np.asarray(result).copy(), cost))
            parent = np.asarray(parent).copy()
            np.testing.assert_array_equal(runs[0][0], runs[1][0])
            np.testing.assert_array_equal(runs[1][0][0], parent[0])
            expected = physical(T, X, roots)
            np.testing.assert_allclose(runs[1][0], expected, rtol=1e-4, atol=2e-6)
            assert np.isfinite(runs[1][0]).all()
            # Uniform metric reproduces all three parent outputs byte-for-byte.
            identity = np.zeros_like(metrics); identity[:, :8, :] = np.eye(8)
            env.update(_ECM8_ROOT=None, _ECM8_ENCODER_LITERAL=identity.tolist())
            uniform = np.asarray(env['_ecm8_energy_contract'](nt, nx)).copy()
            np.testing.assert_array_equal(uniform, parent)
            np.testing.assert_array_equal(np.asarray(nx), X); np.testing.assert_array_equal(np.asarray(nt), T)
            cases.append(dict(n=n, E1_bytes_exact=True, uniform_metric_parent_bytes_exact=True,
                warm_repeat_bytes_exact=True, input_ownership_preserved=True,
                physical_oracle_max_abs=float(np.max(np.abs(runs[1][0]-expected))),
                parent_tariff=parent_cost, cold_tariff=runs[0][1], warm_tariff=runs[1][1]))
        # Exact means and representable rank2 on small positive physical inputs.
        exact_cases = []
        n = 16; T = np.square(rng.normal(size=(3, n, n))).astype(np.float32)
        for rank in (0, 2):
            X = np.broadcast_to(np.linspace(.5, 1, 8), (3, n, 8)).copy()
            if rank: X += .01 * (rng.normal(size=(3, n, 2)) @ rng.normal(size=(3, 2, 8)))
            X = X.astype(np.float32)
            env.update(_ECM8_ROOT=None, _ECM8_ENCODER_LITERAL=metrics.tolist())
            got = np.asarray(env['_ecm8_energy_contract'](fnp.asarray(T), fnp.asarray(X))).copy()
            np.testing.assert_allclose(got, T @ X, rtol=2e-5, atol=3e-5)
            exact_cases.append(dict(source_rank=rank, dense_product_max_abs=float(np.max(np.abs(got-T@X)))))
    n1024 = cases[-1]
    total_debit = (n1024['cold_tariff'][0]-n1024['parent_tariff'][0]) + 14*(n1024['warm_tariff'][0]-n1024['parent_tariff'][0])
    assert total_debit >= 0 and total_debit < 0.00005 * 236000000000
    write(out / 'TINY_OFFICIAL_v1.json', dict(utc=utc(), successful=True, official_version=fs.__version__,
        cases=cases, exact_rank_cases=exact_cases, component_15layer_debit_flops=total_debit,
        approximate_pct_of_236G_reference=100*total_debit/236000000000,
        not_whole_native_bill=True, exact_native_functions=True, no_models=True, no_truth=True))
    subprocess.run(['/home/arcworker/arc-envs/client/bin/python', '-B', str(HERE / 'check_import_v1.py'),
        '--build', str(out/'BUILD_v1.json'), '--out', str(out/'NUMPY_FREE_IMPORT_v1.json')], check=True)
    parentplan = V/'PLAN_BANK8_ACTUAL_SHARED_v1.json'
    assert sha(parentplan) == 'cf76262261a85fcb6b5c508ad37a2b13f92884b6b69aff66bd0da4bcd0833793'
    plan = read(parentplan); science = Path(outputs['science']['path'])
    plan.update(utc=utc(), kind='CURRENT_CM8_ENERGY23_FULL_CONSUMER_METRIC', reference_submission=337422,
        candidates={'energy23_current_metric': dict(path=str(science.relative_to(ROOT)),sha256=sha(science),effective_sha256=sha(science),bytes=science.stat().st_size)})
    planpath = HERE/'PLAN_ENERGY23_CURRENT_METRIC_v1.json';write(planpath,plan)
    ready = dict(utc=utc(),successful=True,outputs=outputs,candidate='energy23_current_metric',
        source_sha256=sha(science),plan=str(planpath),plan_sha256=sha(planpath),
        tiny_sha256=sha(out/'TINY_OFFICIAL_v1.json'),import_sha256=sha(out/'NUMPY_FREE_IMPORT_v1.json'),
        jobs=12,shared_max_active=36,blas_threads=4,reference_submission=337422,
        native_launches=0,fitting=False,truth_access=False,adjusted_score_gain=None)
    write(out/'READY_v1.json',ready);print(json.dumps(ready),flush=True)


if __name__ == '__main__': main()
