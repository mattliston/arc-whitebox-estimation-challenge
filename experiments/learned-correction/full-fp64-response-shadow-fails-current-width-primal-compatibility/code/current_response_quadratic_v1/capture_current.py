"""Inert observation-only current409f context capture; no changed estimator arithmetic."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ENV = NEW.parent/'data_gate/.venv'
BANK = NEW/'unused_materializations_v1/estimator.py'
NATIVE = NEW/'unused_materializations_v1/native'
HELPER = NEW/'check_response_rows_adjusted_v1.py'
PINS = {
    BANK:'409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f',
    NATIVE/'result.json':'a9ccecaa3c0ca79f070af3c0961a6a952f08f4d9712e82a3d6af4d9a0d1d0951',
    HELPER:'8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304',
}
VERSIONS = dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0')
BLAS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
LAYERS = tuple(range(8,15))
RSS_LIMIT = 8_000_000_000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def run():
    assert Path(sys.prefix).resolve()==ENV.resolve() and 'anaconda3' not in sys.executable
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert all(os.environ.get(k)=='1' for k in BLAS)
    assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
    assert {k:importlib.metadata.version(k) for k in VERSIONS}==VERSIONS
    assert all(sha(path)==expected for path,expected in PINS.items())
    assert shutil.disk_usage(HERE).free>=5_000_000_000
    h = module(HELPER,'current_q_capture_utilities')
    approval = json.loads((HERE/'CAPTURE_ADMISSION.json').read_text())
    assert approval['admitted'] and approval['capture_sha256']==sha(Path(__file__))
    assert approval['protocol_sha256']==sha(HERE/'WIDE_PROTOCOL.md')
    assert approval['bank_sha256']==PINS[BANK]
    assert approval['tiny_adjudication_sha256']==sha(HERE/'tiny_affine_adjudication.json')
    tiny = h.load_json(HERE/'tiny_affine_adjudication.json')
    assert tiny['adjudicated_passed'] and not tiny['original_certificate_passed'] and tiny['original_result_unchanged']
    h.check_bindings(tiny['source_bindings'])
    reference = h.load_json(NATIVE/'result.json')
    assert reference['passed'] and reference['hashes_unchanged'] and reference['candidate_sha256']==PINS[BANK]
    fixtures,rows = h.fixture_bindings()
    assert [row['rank'] for row in rows]==[35,100]
    for name,binding in fixtures.items():
        assert binding==reference['fixture_bindings'][name]
    sources = {str(p):h.binding(p) for p in (*PINS,Path(__file__),HERE/'WIDE_PROTOCOL.md',HERE/'CAPTURE_ADMISSION.json',HERE/'tiny_affine_adjudication.json')}
    for row in rows:
        expected = next(t for t in reference['trials'] if t['rank']==row['rank'])
        output_path = NATIVE/expected['output_artifact']
        sources[str(output_path)] = h.binding(output_path)
    directory = HERE/'capture'
    directory.mkdir(exist_ok=False)
    h.write_json(directory/'started.json',dict(utc=h.utc(),pid=os.getpid(),executable=sys.executable,
        prefix=sys.prefix,nice=os.getpriority(os.PRIO_PROCESS,0),versions=VERSIONS,source_bindings=sources,
        fixture_bindings=fixtures,sequence=[35,100],layers=LAYERS,watchdog_s=300,rss_limit_bytes=RSS_LIMIT,
        blas_env={k:os.environ.get(k) for k in BLAS},truth_files_accessed=False,
        scope='Read-only current context capture; no mean/covariance/gate/head changes.'))
    result = dict(passed=False,trials=[],artifacts={},sources=sources,fixture_bindings=fixtures,
                  candidate_sha256=PINS[BANK],truth_files_accessed=False,score_evidence=False)
    timer = time.perf_counter()

    def watchdog(signum,frame):
        raise TimeoutError('Fixed300s capture watchdog')

    signal.signal(signal.SIGALRM,watchdog)
    signal.alarm(300)
    try:
        import numpy as np
        import flopscope as flops
        from threadpoolctl import threadpool_info,threadpool_limits
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        bank = module(BANK,'current_q_observed_parent')
        observed = dict(rank=None,context=None,pair=None,layers=[],pre_layers=[])
        raw_pair,raw_head = bank.nonlinear,bank.gnn_step

        def resources():
            assert h.peak_rss_bytes()<RSS_LIMIT, 'Capture RSS limit exceeded'
            assert time.perf_counter()-timer<300, 'Capture watchdog exceeded'

        def save_arrays(name,values):
            arrays = {k:np.asarray(v) for k,v in values.items()}
            assert all(np.isfinite(v).all() for v in arrays.values()), 'Nonfinite capture'
            path = directory/name
            with path.open('xb') as handle:
                np.savez(handle,**arrays)
            result['artifacts'][name]=h.binding(path)
            resources()

        def context(l,mu,pre,k21,k4,k22,alive,live_indices,response_mask,omitted):
            assert l==len(observed['pre_layers'])
            observed['pre_layers'].append(l)
            observed['pair']=None
            observed['context']=(l,dict(mu=mu,pre_cov=pre,pre_k21=k21,pre_k4=k4,pre_k22=k22,
                alive=alive,live_indices=live_indices,response_mask=response_mask,omitted=omitted))

        def pair(*args,**kwargs):
            answer=raw_pair(*args,**kwargs)
            observed['pair']=answer
            return answer

        def head(l,x,sigma,w,state,payload):
            answer=raw_head(l,x,sigma,w,state,payload)
            if l in LAYERS:
                assert observed['context'][0]==l and observed['pair'] is not None
                host,cov,target,radial,aux=observed['pair']
                values=dict(observed['context'][1],host_mean=host,host_cov=cov,host_k21=target,
                            pair_radial_before_omitted=np.array(radial),features=x,prior_state=state,
                            head_delta=answer[0],next_state=answer[1])
                values.update({'aux_'+k:v for k,v in aux.items() if k!='pre_cov'})
                save_arrays(f'rank{observed["rank"]}_layer{l:02d}.npz',values)
                observed['layers'].append(l)
            observed['pair']=None
            return answer

        tree=ast.parse(BANK.read_text())
        chosen=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_predict_binary'][-1]
        source=ast.get_source_segment(BANK.read_text(),chosen)
        anchor='        m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)'
        assert source.count(anchor)==1
        patched=source.replace(anchor,'        _record_context(l,mu,Cpre,K21,k4,k22,alive,live_indices,response_mask,omitted)\n'+anchor)
        assert patched.replace('        _record_context(l,mu,Cpre,K21,k4,k22,alive,live_indices,response_mask,omitted)\n','')==source
        bank.__dict__['_record_context']=context
        exec(compile(patched,str(BANK)+'::observation_only','exec'),bank.__dict__)
        bank.nonlinear=pair
        bank.gnn_step=head
        estimator=None
        with threadpool_limits(limits=1):
            pools=threadpool_info()
            assert all(p['num_threads']==1 for p in pools if p.get('user_api')=='blas')
            result['threadpools']=pools
            result['blas_attestation']='Environment1; detected pools checked if present, empty inventory is not an observed pool.'
            for call,row in enumerate(rows):
                observed.update(rank=row['rank'],context=None,pair=None,layers=[],pre_layers=[])
                weights=np.load(row['weights_path'],mmap_mode='r',allow_pickle=False)
                assert weights.shape==(16,1024,1024) and weights.dtype==np.float32 and np.isfinite(weights).all()
                mlp=MLP(width=1024,depth=16,weights=list(weights),seed=int(derive_estimator_seed(row['mlp_seed'])),name=f'current-q-capture-{row["rank"]}')
                start=time.perf_counter()
                with flops.BudgetContext(flop_budget=2**41,wall_time_limit_s=120,quiet=True) as ctx:
                    if estimator is None:
                        estimator=bank.Estimator()
                    served=estimator.predict(mlp,2**41)
                    assert isinstance(served,np.ndarray) and served.dtype==np.dtype('float32') and served.shape==(16,1024)
                    output=materialise_predictions(served,depth=16,width=1024)
                elapsed=time.perf_counter()-start
                expected=next(t for t in reference['trials'] if t['rank']==row['rank'])
                prior=np.load(NATIVE/expected['output_artifact'],allow_pickle=False)
                assert hashlib.sha256(prior.tobytes()).hexdigest()==expected['output_bytes_sha256']
                output=np.asarray(output)
                assert served.tobytes()==output.tobytes(), 'Materializer changed served bytes'
                h.assert_output(np,output)
                h.assert_output(np,prior)
                identity=output.dtype==np.float32 and output.shape==(16,1024) and output.tobytes()==prior.tobytes()
                bill=int(ctx.flops_used)
                trial=dict(rank=row['rank'],call=call,flops=bill,reference_flops=expected['flops'],
                    all16_byte_exact=identity,includes_initialization=call==0,context_wall_s=elapsed,
                    meter_wall_s=float(ctx.wall_time_s),backend_s=float(ctx.flopscope_backend_time_s),
                    overhead_s=float(ctx.flopscope_overhead_time_s),
                    diagnostic_residual_s=float(ctx.residual_wall_time_s),rss_bytes=h.peak_rss_bytes(),
                    layers=observed['layers'],all_pre_layers=observed['pre_layers'],
                    output_bytes_sha256=hashlib.sha256(output.tobytes()).hexdigest())
                result['trials'].append(trial)
                print(json.dumps(trial),flush=True)
                assert identity and bill==expected['flops'] and observed['layers']==list(LAYERS)
                assert observed['pre_layers']==list(range(16)) and 0<elapsed<120
                assert 0<trial['meter_wall_s']<120 and 0<bill<=2**41
                assert all(math.isfinite(trial[k]) and trial[k]>=0 for k in ('backend_s','overhead_s','diagnostic_residual_s'))
                resources()
                if call==0:
                    save_arrays('current_P.npz',{k:np.asarray(v) for k,v in bank._GNN_PARAMS.items()})
                del weights,mlp,served,output,prior
        h.check_bindings(sources)
        h.check_bindings(fixtures)
        h.check_bindings(result['artifacts'])
        resources()
        result['passed']=True
    except BaseException:
        result['error']=traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(completed_utc=h.utc(),wall_s=time.perf_counter()-timer,rss_bytes=h.peak_rss_bytes())
    digest=h.write_json(directory/'result.json',result)
    h.write_json(directory/'completion.json',dict(passed=result['passed'],result_sha256=digest))
    print(json.dumps(dict(passed=result['passed'],result_sha256=digest,error=result.get('error'))),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        parser.error('Explicit --run required; inert by default')
    raise SystemExit(run())
