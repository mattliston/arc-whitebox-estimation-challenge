"""Inert fixed synthetic covariance algebra certificate; never a score."""
import argparse
import ast
from dataclasses import fields, replace
from decimal import Decimal
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
Q = NEW/'current_response_quadratic_v1'
S = NEW/'current_response_scalar_certificate_v1'
T = NEW/'current_response_certified_native_v1'
ENV = NEW.parent/'data_gate/.venv'
BANK = NEW/'unused_materializations_v1/estimator.py'
VERSIONS = dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0')
BLAS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
PINS = {
    BANK: '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f',
    Q/'quadratic_covariance_reference.py': 'baa80b74e065e0e974b734a43a2d605de497c369cc32e03de2df1d5bca214527',
    Q/'feature_second_jet_reference.py': 'dd4484fc3baaefb783ff54497b43ff801214e0583cd16d28a9d9b4c212cb218f',
    S/'scalar_interval.py': '80a2cb4ec800648a4c7792fd019f9154f29d43b9dbc8d559b15d0329edc520b0',
    T/'qualification/result.json': '52938606d3f116f5f41021fd19f5f3b03f8b9d7c2713741d0d19cbada4221494',
    T/'gate_information/result.json': 'b549ad49f54d134d4110dcc2c476e2f892b2e3018fb6001958bc50750a7f1f07',
    NEW/'check_response_rows_adjusted_v1.py': '8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304',
    ENV/'lib/python3.11/site-packages/flopscope/stats/_norm.py': 'f23e49201c8e09e2785814b2b31663cda1c6bb217a57a211532e42fc9fe3b65d',
    ENV/'lib/python3.11/site-packages/flopscope/stats/_erf.py': '4cdd75fdb85981fc385417082985478566b18e19fa67ebb7d41225b5efaaa3bd',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def original_fp64_polynomial(np):
    """Only named functions, nine dtype attributes and one assertion string."""
    from flopscope.stats._norm import norm
    names = {'gaussian_raw', 'derivatives', '_nonlinear_pair_batch'}
    chosen = [node for node in ast.parse(BANK.read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(chosen) == 3 and {n.name for n in chosen} == names

    class Dtypes(ast.NodeTransformer):
        attributes = 0
        assertions = 0

        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id='np', ctx=node.ctx), node) if node.id == 'fnp' else node

        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr == 'float32':
                self.attributes += 1
                node.attr = 'float64'
            return node

        def visit_Constant(self, node):
            if node.value == 'float32':
                self.assertions += 1
                return ast.copy_location(ast.Constant(value='float64'), node)
            return node

    transformer = Dtypes()
    tree = ast.fix_missing_locations(transformer.visit(ast.Module(body=chosen, type_ignores=[])))
    assert transformer.attributes == 9 and transformer.assertions == 1
    namespace = dict(np=np, flops=SimpleNamespace(stats=SimpleNamespace(
        norm=SimpleNamespace(cdf=norm._compute_cdf))))
    exec(compile(tree, str(BANK)+'::three_functions_FP64_polynomial', 'exec'), namespace)
    return namespace['_nonlinear_pair_batch']


def run():
    assert Path(sys.prefix).resolve() == ENV.resolve() and 'anaconda3' not in sys.executable
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert all(os.environ.get(k) == '1' for k in BLAS)
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    assert {k: importlib.metadata.version(k) for k in VERSIONS} == VERSIONS
    assert all(sha(p) == expected for p, expected in PINS.items())
    h = module(NEW/'check_response_rows_adjusted_v1.py', 'covariance_algebra_receipts')
    admission = h.load_json(HERE/'ADMISSION.json')
    assert admission['admitted'] and admission['checker_sha256'] == sha(Path(__file__))
    h.check_bindings(admission['source_bindings'])
    required = {str((HERE/n).resolve()) for n in ('PROTOCOL.md', 'check.py')}
    assert required <= {v['path'] for v in admission['source_bindings'].values()}
    prerequisites = {}
    for name in ('qualification', 'gate_information'):
        path = T/name/'result.json'
        value, completion = h.load_json(path), h.load_json(T/name/'completion.json')
        assert value['passed'] and completion['passed'] and completion['result_sha256'] == PINS[path]
        h.check_bindings(value['source_bindings'])
        prerequisites[name] = value
    sources = {str(p): h.binding(p) for p in (*PINS, HERE/'ADMISSION.json')}
    sources.update(admission['source_bindings'])
    directory = HERE/'certificate'
    directory.mkdir(exist_ok=False)
    h.write_json(directory/'started.json', dict(utc=h.utc(), pid=os.getpid(), executable=sys.executable,
        nice=os.getpriority(os.PRIO_PROCESS, 0), versions=VERSIONS, source_bindings=sources,
        blas_env={k: os.environ.get(k) for k in BLAS}, watchdog_s=60, rss_limit_bytes=1_000_000_000,
        synthetic_width=4, truth_arrays_accessed=False, network_arrays_accessed=False, score_evidence=False))
    result = dict(passed=False, source_bindings=sources, records=[], truth_arrays_accessed=False,
                  network_arrays_accessed=False, score_evidence=False, original_failures_unchanged=True)
    timer = time.perf_counter()

    def timeout(signum, frame):
        raise TimeoutError('Fixed60s tiny covariance algebra certificate watchdog')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    try:
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        module(Q/'feature_second_jet_reference.py', 'feature_second_jet_reference')
        q = module(Q/'quadratic_covariance_reference.py', 'fixed_covariance_reference')
        interval = module(S/'scalar_interval.py', 'covariance_independent_intervals')

        def resources():
            assert h.peak_rss_bytes() < 1_000_000_000 and time.perf_counter()-timer < 60

        def record(name, passed, **values):
            row = dict(name=name, passed=bool(passed), **values)
            result['records'].append(row)
            if not passed:
                raise RuntimeError('Fixed algebra guard failed: '+name)
            resources()

        def compare(name, actual, expected):
            actual, expected = np.asarray(actual), np.asarray(expected)
            finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
            error = float(np.max(np.abs(actual-expected))) if finite else None
            magnitude = float(np.max(np.abs(expected))) if finite else None
            record(name, finite and error <= 2e-12+2e-10*magnitude,
                   error_max=error, reference_abs_max=magnitude, absolute=2e-12, relative=2e-10)

        def poly_product(a, b):
            output = [interval.point(0) for _ in range(len(a)+len(b)-1)]
            for i, x in enumerate(a):
                for j, y in enumerate(b):
                    output[i+j] = output[i+j]+x*y
            return output

        def integrate_reference(mu, sigma, a, b, order):
            mu, sigma, a, b = map(interval.point, (mu, sigma, a, b))
            qpoly = [-b*sigma.square(), a*sigma, b*sigma.square()]
            hermite = {0: [1], 3: [0, -3, 0, 1], 4: [3, 0, -6, 0, 1]}[order]
            hp = [interval.point(c) for c in hermite]
            cross = poly_product(poly_product([mu, sigma], qpoly), hp)
            square = poly_product(poly_product(qpoly, qpoly), hp)
            t = -mu/sigma
            density = interval.normal_pdf(t)
            tail = [1-interval.normal_cdf(t), density]
            for k in range(2, max(len(cross), len(square))):
                tail.append(t**(k-1)*density+(k-1)*tail[k-2])
            full = [interval.point(1), interval.point(0)]
            for k in range(2, max(len(cross), len(square))):
                full.append((k-1)*full[k-2])
            value = interval.point(0)
            for k, c in enumerate(cross):
                value = value+2*c*tail[k]
            for k, c in enumerate(square):
                value = value+c*full[k]
            return value/(sigma**order)

        def compare_interval(name, actual, reference):
            difference = abs(interval.point(float(actual))-reference)
            magnitude = abs(reference)
            allowance = interval.point(Decimal('2e-12'))+interval.point(Decimal('2e-10'))*magnitude
            record(name, difference.hi <= allowance.lo, actual=float(actual),
                   reference=reference.as_strings(), error_upper=str(difference.hi),
                   allowance_lower=str(allowance.lo))

        def snapshots(reference):
            return {f.name: getattr(reference, f.name).tobytes() for f in fields(reference)
                    if isinstance(getattr(reference, f.name), np.ndarray)} | {
                        f'{key}{i}': a.tobytes() for key in ('A', 'H')
                        for i, a in enumerate(getattr(reference, key))}

        with threadpool_limits(limits=1):
            pools = threadpool_info()
            assert all(p['num_threads'] == 1 for p in pools if p.get('user_api') == 'blas')
            result['threadpools'] = pools
            result['blas_attestation'] = 'Environment1; any detected pools checked. Empty inventory is not observed pool evidence.'
            mu = np.array([-.375, 0., .75, 1.5])
            sigma = np.array([.5, 1., 1.5, 2.])
            variance = sigma*sigma
            correlation = np.array([[1, 1/8, -1/16, 1/32], [1/8, 1, 1/16, -1/32],
                                    [-1/16, 1/16, 1, 1/8], [1/32, -1/32, 1/8, 1]])
            cov = np.outer(sigma, sigma)*correlation
            k21 = np.outer(variance, sigma)*np.array(
                [[64, 1, -2, 3], [-4, 32, 2, 1], [2, -3, -32, 4], [1, 2, -1, 16]])/4096
            k4 = variance*variance*np.array([1, -2, 3, -1])/256
            k22 = np.outer(variance, variance)*np.array(
                [[32, 1, -1, 2], [1, -64, 2, -2], [-1, 2, 96, 1], [2, -2, 1, -32]])/8192
            a, b = np.array([1., -2., 3., -1.])/1024, np.array([2., 1., -1., 3.])/4096
            inputs = dict(mu=mu, sigma=sigma, cov=cov, k21=k21, k4=k4, k22=k22, a=a, b=b)
            initial_inputs = {k: v.tobytes() for k, v in inputs.items()}
            result['fixture'] = {k: v.tolist() for k, v in inputs.items()}
            reference = q.prepare_reference(mu, cov, k21, k4, k22)
            record('exact_fixture_sigma', np.array_equal(reference.sigma, sigma))
            initial_reference = snapshots(reference)
            mask = ~np.eye(4, dtype=bool)
            native64 = original_fp64_polynomial(np)(mu, cov, k21, k4, k22, 'covariance')
            compare('full_null_vs_independent_AST_offdiagonal',
                    q.full_pair11_reference(reference)[mask], native64[1][mask])
            zero = np.zeros(4)
            operators = ('t21', 't22', 't31', 't3', 't4')
            empty = {name: np.zeros_like(getattr(reference, name)) for name in operators}
            empty.update(pre_k3=zero.copy(), pre_k4=zero.copy())
            gaussian = replace(reference, **empty)
            configurations = [('gaussian', gaussian)]
            for name in operators:
                supplied = dict(empty)
                supplied[name] = getattr(reference, name)
                if name == 't3':
                    supplied['pre_k3'] = reference.pre_k3
                if name == 't4':
                    supplied['pre_k4'] = reference.pre_k4
                configurations.append((name, replace(reference, **supplied)))
            configurations.append(('full', reference))
            cases = [('null', zero, zero), ('a_only', a, zero), ('b_only', zero, b), ('both', a, b)]
            for operator_name, operator in configurations:
                initial_operator = snapshots(operator)
                base = q.full_pair11_reference(operator)
                for case, ca, cb in cases:
                    delta = q.covariance_delta(operator, ca, cb)
                    dense = q.full_pair11_reference(operator, ca, cb)-base
                    prefix = operator_name+'/'+case
                    compare(prefix+'/sparse_vs_dense_offdiagonal', delta['covariance'][mask], dense[mask])
                    record(prefix+'/finite_zero_mean', np.isfinite(delta['covariance']).all()
                           and np.count_nonzero(delta['mean_delta']) == 0)
                    compare(prefix+'/symmetry', delta['covariance'], delta['covariance'].T)
                    if case == 'null':
                        record(prefix+'/exact_null', np.count_nonzero(delta['covariance']) == 0)
                    if operator_name == 'gaussian':
                        g, d = reference.A[1], reference.A[2]/sigma
                        closed = cov*(np.outer(g, ca)+np.outer(ca, g)+np.outer(ca, ca))
                        closed += cov*cov*(np.outer(d, cb)+np.outer(cb, d)+2*np.outer(cb, cb))
                        compare(prefix+'/independent_Gaussian_offdiagonal', delta['covariance'][mask], closed[mask])
                    if operator_name == 'full':
                        for j in range(4):
                            raw = {r: integrate_reference(float(mu[j]), float(sigma[j]),
                                float(ca[j]), float(cb[j]), r) for r in (0, 3, 4)}
                            for r in (0, 3, 4):
                                compare_interval(prefix+f'/unit{j}/raw{r}',
                                    delta['marginal_raw_second_derivative_deltas'][r][j], raw[r])
                            diagonal = raw[0]+interval.point(float(k21[j, j]))*raw[3]/6
                            diagonal = diagonal+interval.point(float(k4[j]))*raw[4]/24
                            compare_interval(prefix+f'/unit{j}/assembled_diagonal', delta['diagonal_delta'][j], diagonal)
                record(operator_name+'/reference_unchanged', initial_operator == snapshots(operator))
            low = cov.copy()
            low[0, 0] = 1e-11
            low_reference = q.prepare_reference(mu, low, k21, k4, k22)
            rejected = False
            try:
                q.covariance_delta(low_reference, a, b)
            except ValueError as error:
                rejected = 'Below-sigma-floor' in str(error)
            record('below_floor_action_rejected', rejected and low_reference.below_sigma_floor_count == 1)
            for value in (0., -.25):
                bad = cov.copy()
                bad[0, 0] = value
                rejected = False
                try:
                    q.prepare_reference(mu, bad, k21, k4, k22)
                except ValueError as error:
                    rejected = 'Nonpositive raw variance' in str(error)
                record('nonpositive_raw_variance_rejected/'+str(value), rejected)
            record('all_original_inputs_unchanged', initial_inputs == {k: v.tobytes() for k, v in inputs.items()})
            record('full_original_reference_unchanged', initial_reference == snapshots(reference))
            h.check_bindings(sources)
            for value in prerequisites.values():
                h.check_bindings(value['source_bindings'])
            resources()
            result['passed'] = all(r['passed'] for r in result['records'])
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(completed_utc=h.utc(), wall_s=time.perf_counter()-timer, rss_bytes=h.peak_rss_bytes(),
        qualification='Fixed synthetic polynomial/interval algebra only. No native participant parity, current covariance quality, complete cost, final raw MSE or adjusted-score evidence.')
    seal = h.write_json(directory/'result.json', result)
    h.write_json(directory/'completion.json', dict(passed=result['passed'], result_sha256=seal))
    print(json.dumps(dict(passed=result['passed'], result_sha256=seal, checks=len(result['records']),
                         wall_s=result['wall_s'], rss_bytes=result['rss_bytes'], error=result.get('error'))), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('Explicit --run and independent admission required; inert by default')
    raise SystemExit(run())
