"""Explicit-run synthetic qualification; no estimator/MLP/truth/payload imports."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ENV = NEW.parent / 'data_gate/.venv'
BANK = NEW / 'unused_materializations_v1/estimator.py'
BANK_SHA = '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
VERSIONS = dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0')
BLAS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def independent_formulas(np, ndtr):
    """AST-only source extraction; no payload statement or estimator code executes."""
    tree = ast.parse(BANK.read_text())
    names = {'gaussian_raw', 'derivatives', '_nonlinear_pair_batch', 'features', 'gnn_step', 'silu'}
    chosen = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    assert set(chosen) == names

    class Extension(ast.NodeTransformer):
        def __init__(self, function):
            self.function = function
            self.dtype_assertions = 0

        def visit_Assert(self, node):
            node = self.generic_visit(node)
            if self.function == '_nonlinear_pair_batch':
                expected = ast.parse("str(ans.dtype)==str(rr.dtype)==str(ec.dtype)=='float32'", mode='eval').body
                assert ast.dump(node.test) == ast.dump(expected), 'Unexpected source assertion'
                node.test.comparators[-1].value = 'float64'
                self.dtype_assertions += 1
            return node

        def visit_Name(self, node):
            if node.id == 'fnp':
                return ast.copy_location(ast.Name(id='np', ctx=node.ctx), node)
            return node

        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if self.function == '_nonlinear_pair_batch' and isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr == 'float32':
                node.attr = 'float64'
            return node

        def visit_Return(self, node):
            node = self.generic_visit(node)
            if self.function == 'features' and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'astype':
                node.value = node.value.func.value
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if self.function == 'gnn_step' and isinstance(node.func, ast.Attribute) and node.func.attr == 'astype' and isinstance(node.func.value, ast.Name) and node.func.value.id in ('x', 'sigma'):
                node.args = [ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64', ctx=ast.Load())]
            return node

    transformed = []
    for name in sorted(names):
        extension = Extension(name)
        transformed.append(extension.visit(chosen[name]))
        assert extension.dtype_assertions == (1 if name == '_nonlinear_pair_batch' else 0)
    module = ast.fix_missing_locations(ast.Module(body=transformed, type_ignores=[]))
    env = dict(np=np, flops=SimpleNamespace(stats=SimpleNamespace(norm=SimpleNamespace(cdf=ndtr))))
    exec(compile(module, str(BANK)+'::continuous_extension', 'exec'), env)
    return env


def run():
    assert Path(sys.prefix).resolve() == ENV.resolve() and 'anaconda3' not in sys.executable
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert all(os.environ.get(k) == '1' for k in BLAS)
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    assert {k: importlib.metadata.version(k) for k in VERSIONS} == VERSIONS
    assert digest(BANK) == BANK_SHA
    assert not (HERE/'tiny_result.json').exists() and not (HERE/'tiny_started.json').exists()
    sources = {str(p): digest(p) for p in (BANK, HERE/'PROTOCOL.md', Path(__file__),
               HERE/'feature_second_jet_reference.py', HERE/'gnn_second_jet_reference.py')}
    started = dict(utc=utc(), pid=os.getpid(), executable=sys.executable, prefix=sys.prefix,
                   versions=VERSIONS, source_sha256=sources, nice=os.getpriority(os.PRIO_PROCESS,0),
                   blas_env={k: os.environ.get(k) for k in BLAS}, watchdog_s=60,
                   peak_rss_limit_bytes=1_000_000_000, targets_opened=False,
                   mlp_or_payload_arrays_opened=False, score_evidence=False)
    save(HERE/'tiny_started.json', started)
    timer = time.perf_counter()
    result = dict(passed=False, started=started, records=[])

    def timeout(signum, frame):
        raise TimeoutError('Fixed synthetic60s watchdog')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    try:
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        import feature_second_jet_reference as features
        import gnn_second_jet_reference as gnn
        env = independent_formulas(np, features.normal_cdf)
        rng = np.random.Generator(np.random.PCG64DXSM(0x514a4554))
        n = 16
        a = rng.normal(size=(n,n))/np.sqrt(n)
        gram = a@a.T + np.eye(n)
        corr = gram/np.sqrt(np.outer(np.diag(gram),np.diag(gram)))
        sigma0 = rng.uniform(.6,1.2,size=n)
        alpha = rng.uniform(-1.7,1.7,size=n)
        w = (rng.normal(size=(n,n))*.06).astype(np.float32)
        k21_shape = rng.normal(size=(n,n))*.003
        k4_shape = rng.normal(size=n)*.006
        k22_shape = rng.normal(size=(n,n))*.003
        k22_shape = (k22_shape+k22_shape.T)/2
        payload = {name: (rng.normal(size=shape)*.035).astype(np.float32)
                   for name, shape in gnn._HEAD_SHAPES.items()}
        payload['centers'] = np.zeros((16,30),np.float32)
        payload['scales'] = np.ones((16,30),np.float32)
        payload['rms'] = np.full(16,.01,np.float32)
        payload['msg_w'] = (rng.normal(size=(48,64))*.035).astype(np.float32)
        payload['msg_b'] = (rng.normal(size=48)*.035).astype(np.float32)
        prior_state = (rng.normal(size=(n,64))*.1).astype(np.float32)
        extended_payload = {k: v if k in ('msg_w','msg_b') else v.astype(np.float64)
                            for k,v in payload.items()}

        def comparison(actual, expected, tolerance, absolute=0, zero_reference=False):
            norm = float(np.linalg.norm(expected))
            error = float(np.linalg.norm(actual-expected))
            allowed_absolute = absolute if zero_reference and not np.count_nonzero(actual) else 0.
            return dict(error_norm=error, reference_norm=norm,
                        relative_error=error/max(norm,1e-300),
                        allowed_absolute=allowed_absolute,
                        passed=bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                                    and error <= allowed_absolute+tolerance*norm))

        fixed_features = {0,1,2,4,10,12,13,14,15,16,17,18,19,22,23,24,25}

        def feature_comparison(actual, expected, tolerance, absolute):
            records = [dict(feature=k, **comparison(actual[:,k],expected[:,k],tolerance,absolute,
                                                   zero_reference=k in fixed_features)) for k in range(30)]
            return dict(passed=all(r['passed'] for r in records), per_feature=records)

        with threadpool_limits(limits=1):
            pools = threadpool_info()
            assert all(p['num_threads']==1 for p in pools if p.get('user_api')=='blas')
            result['threadpools'] = pools
            result['blas_attestation'] = 'Environment1; detected BLAS pools verified when present. Empty inventory is not observed pool evidence.'
            fixed_messages = gnn.prepare_fixed_messages(w,prior_state,payload)
            prepared_head = gnn.prepare_head(payload)
            src = prior_state@payload['msg_w'].T+payload['msg_b']
            w32 = w.astype(np.float32)
            w2,w3 = w32*w32,(w32*w32)*w32
            m1 = w32.T@src[:,:16]
            m2 = w2.T@src[:,16:32]
            m3 = w3.T@src[:,32:]
            e1 = w2.T@(src[:,:16]**2)
            e2 = (w2*w2).T@(src[:,16:32]**2)
            e3 = (w3*w3).T@(src[:,32:]**2)
            independent_messages = (m1,m2,m3,np.log1p(np.maximum(e1,0)),
                                     np.log1p(np.maximum(e2,0)),np.log1p(np.maximum(e3,0)))
            result['fixed_messages_byte_exact'] = [bool(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes())
                                                   for a,b in zip(fixed_messages.values,independent_messages)]
            for name in ('regular','below_sigma_floor'):
                sig = sigma0.copy()
                if name == 'below_sigma_floor':
                    sig[0] = 5e-6
                pre = corr*np.outer(sig,sig)
                sigma = np.sqrt(np.maximum(np.diag(pre),1e-10))
                mu = alpha*sigma
                k21 = k21_shape*(sig[:,None]**2)*sig[None,:]
                k4 = k4_shape*sig**4
                k22 = k22_shape*np.outer(sig**2,sig**2)
                jet, aux, covariance, boundaries = features.features(12,mu,pre,k21,k4,k22,w)
                host = env['_nonlinear_pair_batch'](mu,pre,k21,k4,k22,'covariance')
                independent_x = env['features'](12,host[0],host[1],k21,host[4],w)
                response = gnn.step(12,jet.value,jet.first,jet.second,sigma,w,prior_state,payload,
                                    messages=fixed_messages,head=prepared_head)
                independent_delta, independent_state = env['gnn_step'](12,independent_x,sigma,w,prior_state,extended_payload)
                rec = dict(case=name, boundaries=boundaries, primal={
                    'features': comparison(jet.value,independent_x,2e-12),
                    'covariance': comparison(covariance.value,host[1],2e-12),
                    'mean': comparison(aux['mean'].value,host[0],2e-12),
                    'gnn_delta': comparison(response.delta,independent_delta,2e-12),
                    'gnn_state': comparison(response.state,independent_state,2e-12)}, differences=[])
                rec['diagnostics'] = response.diagnostics
                rec['gnn_reference_ledger_flops'] = response.cost['total_flops']
                rec['fixed_message_preparation_flops'] = fixed_messages.cost['total_flops']
                rec['head_preparation_flops'] = prepared_head.cost['total_flops']
                for eps_factor in (.001,.0005):
                    ids = np.array([0,5,10,15])
                    fd1,fd2,gd1,gd2 = [],[],[],[]
                    for j in ids:
                        epsilon = eps_factor*sigma[j]
                        mp,mm = mu.copy(),mu.copy()
                        mp[j] += epsilon
                        mm[j] -= epsilon
                        op = env['_nonlinear_pair_batch'](mp,pre,k21,k4,k22,'covariance')
                        om = env['_nonlinear_pair_batch'](mm,pre,k21,k4,k22,'covariance')
                        xp = env['features'](12,op[0],op[1],k21,op[4],w)
                        xm = env['features'](12,om[0],om[1],k21,om[4],w)
                        dp = env['gnn_step'](12,xp,sigma,w,prior_state,extended_payload)[0]
                        dm = env['gnn_step'](12,xm,sigma,w,prior_state,extended_payload)[0]
                        fd1.append((xp[j]-xm[j])/(2*epsilon))
                        fd2.append((xp[j]-2*independent_x[j]+xm[j])/(epsilon*epsilon))
                        gd1.append((dp[j]-dm[j])/(2*epsilon))
                        gd2.append((dp[j]-2*independent_delta[j]+dm[j])/(epsilon*epsilon))
                    dimension1,dimension2 = sigma[ids],sigma[ids]**2
                    rec['differences'].append(dict(epsilon_factor=eps_factor,units=ids.tolist(),
                        feature_first=feature_comparison(jet.first[ids]*dimension1[:,None],np.array(fd1)*dimension1[:,None],2e-6,1e-10),
                        feature_second=feature_comparison(jet.second[ids]*dimension2[:,None],np.array(fd2)*dimension2[:,None],5e-4,1e-8),
                        gnn_first=comparison(response.ddelta[ids]*dimension1,np.array(gd1)*dimension1,2e-6),
                        gnn_second=comparison(response.d2delta[ids]*dimension2,np.array(gd2)*dimension2,5e-4)))
                rec['passed'] = all(v['passed'] for v in rec['primal'].values()) and all(
                    all(d[k]['passed'] for k in ('feature_first','feature_second','gnn_first','gnn_second'))
                    for d in rec['differences']) and response.diagnostics['all_finite']
                result['records'].append(rec)
                print(json.dumps(dict(case=name,passed=rec['passed'],primal=rec['primal'],
                    differences=[dict(epsilon_factor=d['epsilon_factor'],gnn_first=d['gnn_first'],gnn_second=d['gnn_second'],
                                      failed_feature_first=[r for r in d['feature_first']['per_feature'] if not r['passed']],
                                      failed_feature_second=[r for r in d['feature_second']['per_feature'] if not r['passed']])
                                 for d in rec['differences']])),flush=True)
            # Finite equality cases are algebraic conventions, not FD claims.
            x = features.Jet(np.array([-9.,-8.,0.,8.,9.]),np.ones(5),np.ones(5)*2)
            clipped = features.clip(x,-8,8)
            result['clip_convention_passed'] = bool(np.array_equal(clipped.first,[0,0,1,0,0]) and np.array_equal(clipped.second,[0,0,2,0,0]))
            gnn_clipped = gnn._clip_jet(x.value,x.first,x.second,gnn.CostLedger(),'explicit_boundary',
                                       dict(clip_endpoint_counts={},nonfinite={}))
            result['gnn_clip_convention_passed'] = bool(np.array_equal(gnn_clipped[1],[0,0,1,0,0]) and np.array_equal(gnn_clipped[2],[0,0,2,0,0]))
            result['passed'] = (all(r['passed'] for r in result['records']) and result['clip_convention_passed']
                                and result['gnn_clip_convention_passed'] and all(result['fixed_messages_byte_exact']))
        assert all(digest(Path(p))==sha for p,sha in sources.items()), 'Sources changed during certificate'
    except BaseException:
        result['error'] = traceback.format_exc()
        result['passed'] = False
    finally:
        signal.alarm(0)
    result['wall_s'] = time.perf_counter()-timer
    result['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['resources_passed'] = result['wall_s']<60 and result['peak_rss_bytes']<1_000_000_000
    result['passed'] = result['passed'] and result['resources_passed']
    result['completed_utc'] = utc()
    save(HERE/'tiny_result.json',result)
    print(json.dumps(dict(passed=result['passed'],wall_s=result['wall_s'],rss=result['peak_rss_bytes'],error=result.get('error'),result_sha256=digest(HERE/'tiny_result.json'))),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('Explicit --run required; inert by default')
    raise SystemExit(run())
