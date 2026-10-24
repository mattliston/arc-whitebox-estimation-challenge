"""Explicit-run target-blind current-state response qualification, not a score."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ENV = NEW.parent/'data_gate/.venv'
CAPTURE_SHA = '575727d6f465cbdfeeb9f58d201e9a6e2983aef94dcd2d2389108c58fe569b42'
PINS = {
    HERE/'feature_second_jet_reference.py':'dd4484fc3baaefb783ff54497b43ff801214e0583cd16d28a9d9b4c212cb218f',
    HERE/'gnn_second_jet_reference.py':'2ce74c478040d456078091751520d3a845b2fc8ca47acd6269d8fd8d255d8b21',
    HERE/'certify_tiny.py':'bd8594f6a6138f75996f9ac764e062c684a30a68a18de5ec91f6d224a31e4ab7',
    HERE/'WIDE_PROTOCOL.md':'66612ea988711b093f926807ebf05958839c3d18fbf7857929071fa613bba12e',
    NEW/'check_response_rows_adjusted_v1.py':'8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304',
}
VERSIONS = dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0')
BLAS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    assert Path(sys.prefix).resolve()==ENV.resolve() and 'anaconda3' not in sys.executable
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert all(os.environ.get(k)=='1' for k in BLAS)
    assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
    assert {k:importlib.metadata.version(k) for k in VERSIONS}==VERSIONS
    assert all(sha(p)==digest for p,digest in PINS.items())
    spec=importlib.util.spec_from_file_location('current_wide_helper',NEW/'check_response_rows_adjusted_v1.py')
    h=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    approval=h.load_json(HERE/'WIDE_ADMISSION.json')
    assert approval['admitted'] and approval['checker_sha256']==sha(Path(__file__))
    assert approval['capture_sha256']==CAPTURE_SHA
    assert sha(HERE/'capture/result.json')==CAPTURE_SHA
    capture=h.load_json(HERE/'capture/result.json')
    completion=h.load_json(HERE/'capture/completion.json')
    assert capture['passed'] and completion['passed'] and completion['result_sha256']==CAPTURE_SHA
    assert [t['rank'] for t in capture['trials']]==[35,100]
    assert all(t['all16_byte_exact'] and t['flops']==t['reference_flops'] for t in capture['trials'])
    for field in ('sources','fixture_bindings','artifacts'):
        h.check_bindings(capture[field])
    sources={str(p):h.binding(p) for p in (*PINS,Path(__file__),HERE/'WIDE_ADMISSION.json',HERE/'capture/result.json')}
    directory=HERE/'wide'
    directory.mkdir(exist_ok=False)
    h.write_json(directory/'started.json',dict(utc=h.utc(),pid=os.getpid(),executable=sys.executable,
        prefix=sys.prefix,nice=os.getpriority(os.PRIO_PROCESS,0),versions=VERSIONS,sources=sources,
        capture_sha256=CAPTURE_SHA,blas_env={k:os.environ.get(k) for k in BLAS},
        watchdog_s=180,rss_limit_bytes=2_000_000_000,truth_files_accessed=False,
        score_evidence=False,served_fp32_derivative_claim=False))
    result=dict(passed=False,records=[],artifacts={},source_bindings=sources,capture_sha256=CAPTURE_SHA,
                truth_files_accessed=False,score_evidence=False,served_fp32_derivative_claim=False)
    timer=time.perf_counter()

    def watchdog(signum,frame):
        raise TimeoutError('Fixed180s wide qualifier watchdog')

    signal.signal(signal.SIGALRM,watchdog)
    signal.alarm(180)
    try:
        import numpy as np
        from threadpoolctl import threadpool_info,threadpool_limits
        import feature_second_jet_reference as feat
        import gnn_second_jet_reference as gnn
        import certify_tiny as tiny
        env=tiny.independent_formulas(np,feat.normal_cdf)
        payload=dict(np.load(HERE/'capture/current_P.npz',allow_pickle=False))
        assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in payload.values())
        extended_payload={k:v if k in ('msg_w','msg_b') else v.astype(np.float64) for k,v in payload.items()}
        prepared_head=gnn.prepare_head(payload)

        def resources():
            assert h.peak_rss_bytes()<2_000_000_000
            assert time.perf_counter()-timer<180

        def relative(actual,expected,tolerance,absolute_max=0.):
            difference=actual-expected
            norm=float(np.linalg.norm(expected))
            error=float(np.linalg.norm(difference))
            maximum=float(np.max(np.abs(difference)))
            return dict(error_norm=error,reference_norm=norm,relative_error=error/max(norm,1e-300),
                        max_absolute=maximum,passed=bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                        and (error<=tolerance*norm or maximum<=absolute_max)))

        def mixed(actual,expected,tolerance,absolute):
            norm=float(np.linalg.norm(expected))
            error=float(np.linalg.norm(actual-expected))
            return dict(error_norm=error,reference_norm=norm,relative_error=error/max(norm,1e-300),
                        tolerance=tolerance,absolute=absolute,
                        passed=bool(np.isfinite(actual).all() and np.isfinite(expected).all() and error<=absolute+tolerance*norm))

        with threadpool_limits(limits=1):
            pools=threadpool_info()
            assert all(p['num_threads']==1 for p in pools if p.get('user_api')=='blas')
            result['threadpools']=pools
            result['blas_attestation']='Environment1; any detected BLAS pools checked; empty list is not observed pool evidence.'
            for rank in (35,100):
                weights=np.load(capture['fixture_bindings'][f'weights_{rank}']['path'],mmap_mode='r',allow_pickle=False)
                saved={k:[] for k in ('gaussian_gate','host_response','learned_response','delta_derivative','delta_second_derivative','host_derivative','host_second_derivative')}
                for layer in range(8,15):
                    frame=dict(np.load(HERE/f'capture/rank{rank}_layer{layer:02d}.npz',allow_pickle=False))
                    mu,pre,k21,k4,k22=(frame[k] for k in ('mu','pre_cov','pre_k21','pre_k4','pre_k22'))
                    assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in (mu,pre,k21,k4,k22))
                    w=np.asarray(weights[layer])
                    state=frame['prior_state']
                    jet,aux,cov,boundaries=feat.features(layer,mu,pre,k21,k4,k22,w)
                    sigma=aux['sigma']
                    messages=gnn.prepare_fixed_messages(w,state,payload)
                    response=gnn.step(layer,jet.value,jet.first,jet.second,sigma,w,state,payload,messages=messages,head=prepared_head)
                    feature_value=[dict(feature=k,**relative(jet.value[:,k],frame['features'][:,k].astype(np.float64),1e-6,1e-8)) for k in range(30)]
                    delta_discrepancy=float(np.max(np.abs(response.delta-frame['head_delta'])/sigma))
                    checks=dict(host_mean=relative(aux['mean'].value,frame['host_mean'],1e-12),
                                sigma=relative(sigma,frame['aux_sigma'],1e-12),
                                state=relative(response.state,frame['next_state'].astype(np.float64),1e-5),
                                features_passed=all(t['passed'] for t in feature_value),
                                delta_max_sigma_scaled_discrepancy=delta_discrepancy,
                                delta_passed=delta_discrepancy<=1e-7)
                    moving_boundaries=boundaries['post_variance_floor']+boundaries['diagonal_lower_clip']+boundaries['offdiagonal_post_clip']
                    moving_boundaries+=sum(response.diagnostics['clip_endpoint_counts'].values())
                    rec=dict(rank=rank,layer=layer,value=checks,feature_value=feature_value,
                        boundaries=boundaries,moving_boundary_count=moving_boundaries,
                        gnn_diagnostics=response.diagnostics,finite_difference=[],
                        reference_gnn_jet_flops_subtotal=response.cost['total_flops'],
                        reference_fixed_message_flops_subtotal=messages.cost['total_flops'],
                        head_preparation_flops_subtotal=prepared_head.cost['total_flops'],
                        cost_qualification='Offline analytical GNN subtotal, not complete new-estimator or diagnostic bill.')
                    host=aux['mean'].first
                    learned=host+response.ddelta
                    rec['response_magnitudes']=dict(delta_first_rms=float(np.sqrt(np.mean(response.ddelta**2))),
                        delta_second_sigma_rms=float(np.sqrt(np.mean((response.d2delta*sigma)**2))),
                        host_clipped=int(np.count_nonzero((host<0)|(host>1))),
                        learned_clipped=int(np.count_nonzero((learned<0)|(learned>1))))
                    if layer in (8,14):
                        original=env['_nonlinear_pair_batch'](mu,pre,k21,k4,k22,'covariance')
                        x0=env['features'](layer,original[0],original[1],k21,original[4],w)
                        d0=env['gnn_step'](layer,x0,sigma,w,state,extended_payload)[0]
                        rec['independent_shadow_value']=dict(features=relative(jet.value,x0,2e-12),delta=relative(response.delta,d0,2e-12))
                        ids=np.array([0,257,514,771])
                        for factor in (.001,.0005):
                            first,second,gfirst,gsecond=[],[],[],[]
                            for j in ids:
                                epsilon=factor*sigma[j]
                                plus,minus=mu.copy(),mu.copy()
                                plus[j]+=epsilon
                                minus[j]-=epsilon
                                op=env['_nonlinear_pair_batch'](plus,pre,k21,k4,k22,'covariance')
                                om=env['_nonlinear_pair_batch'](minus,pre,k21,k4,k22,'covariance')
                                xp=env['features'](layer,op[0],op[1],k21,op[4],w)
                                xm=env['features'](layer,om[0],om[1],k21,om[4],w)
                                dp=env['gnn_step'](layer,xp,sigma,w,state,extended_payload)[0]
                                dm=env['gnn_step'](layer,xm,sigma,w,state,extended_payload)[0]
                                first.append((xp[j]-xm[j])/(2*epsilon))
                                second.append((xp[j]-2*x0[j]+xm[j])/(epsilon*epsilon))
                                gfirst.append((dp[j]-dm[j])/(2*epsilon))
                                gsecond.append((dp[j]-2*d0[j]+dm[j])/(epsilon*epsilon))
                            first,second=np.array(first),np.array(second)
                            ff=[dict(feature=k,**mixed(jet.first[ids,k]*sigma[ids],first[:,k]*sigma[ids],2e-6,1e-8)) for k in range(30)]
                            fs=[dict(feature=k,**mixed(jet.second[ids,k]*sigma[ids]**2,second[:,k]*sigma[ids]**2,5e-4,1e-6)) for k in range(30)]
                            gf=mixed(response.ddelta[ids],np.array(gfirst),1e-4,1e-9)
                            gs=mixed(response.d2delta[ids]*sigma[ids],np.array(gsecond)*sigma[ids],5e-4,1e-8)
                            rec['finite_difference'].append(dict(epsilon_factor=factor,units=ids.tolist(),feature_first=ff,feature_second=fs,
                                gnn_first=gf,gnn_second=gs,passed=all(t['passed'] for t in ff+fs) and gf['passed'] and gs['passed']))
                            resources()
                    primal=checks['host_mean']['passed'] and checks['sigma']['passed'] and checks['state']['passed'] and checks['features_passed'] and checks['delta_passed']
                    rec['passed']=bool(primal and moving_boundaries==0 and response.diagnostics['all_finite']
                        and response.diagnostics['nonpositive_head_scales']==0
                        and all(t['passed'] for t in rec['finite_difference'])
                        and all(t['passed'] for t in rec.get('independent_shadow_value',{}).values()))
                    result['records'].append(rec)
                    print(json.dumps(dict(rank=rank,layer=layer,passed=rec['passed'],value=checks,moving_boundaries=moving_boundaries,
                        response=rec['response_magnitudes'],fd_passes=[t['passed'] for t in rec['finite_difference']])),flush=True)
                    saved['gaussian_gate'].append(aux['gate'].value.copy())
                    saved['host_response'].append(np.clip(host,0,1))
                    saved['learned_response'].append(np.clip(learned,0,1))
                    saved['delta_derivative'].append(response.ddelta.copy())
                    saved['delta_second_derivative'].append(response.d2delta.copy())
                    saved['host_derivative'].append(host.copy())
                    saved['host_second_derivative'].append(aux['mean'].second.copy())
                    resources()
                    if not rec['passed']:
                        raise RuntimeError(f'Fixed wide qualification failed rank{rank}/layer{layer}; no further data/targets.')
                    del frame,jet,aux,cov,response,messages
                name=f'rank{rank}_responses.npz'
                with (directory/name).open('xb') as handle:
                    np.savez(handle,**{k:np.stack(v) for k,v in saved.items()},layers=np.arange(8,15))
                result['artifacts'][name]=h.binding(directory/name)
                del weights,saved
        for field in ('sources','fixture_bindings','artifacts'):
            h.check_bindings(capture[field])
        h.check_bindings(sources)
        h.check_bindings(result['artifacts'])
        resources()
        result['passed']=len(result['records'])==14
    except BaseException:
        result['error']=traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(completed_utc=h.utc(),wall_s=time.perf_counter()-timer,rss_bytes=h.peak_rss_bytes())
    seal=h.write_json(directory/'result.json',result)
    h.write_json(directory/'completion.json',dict(passed=result['passed'],result_sha256=seal))
    print(json.dumps(dict(passed=result['passed'],result_sha256=seal,wall_s=result['wall_s'],rss_bytes=result['rss_bytes'],error=result.get('error'))),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        parser.error('Explicit --run required; inert by default')
    raise SystemExit(run())
