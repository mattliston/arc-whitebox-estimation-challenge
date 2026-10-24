"""One fixed CPU execution qualifier, not a fit or native-score estimate."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import weakref

HERE=Path(__file__).resolve().parent
SUPERVISOR_OWNER=HERE.parent/'current_native_trajectory_adjoint_v1/supervise.py'
SUPERVISOR_SHA='648e5227795a8d2e5a91a1a421e0e54ffea1f76d6d6d65b28335178c8772025d'
NAMES={'qualification.py','QUALIFICATION_PROTOCOL.md','run_qualification.py','supervise.py'}
LIMITS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true');ap.add_argument('--pins')
    args=ap.parse_args()
    if not args.run:ap.print_help();return 0
    began=time.monotonic();output=HERE/'qualification_v1'
    require(not output.exists(),'Exclusive fresh qualifier, never resume/overwrite');output.mkdir()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Whole qualifier120s')))
    signal.alarm(120)
    result=dict(passed=False,completed=False,truth_accessed=False,fits=0,full_model_calls=0,
        backwards=0,parameter_updates=0,optimizer_steps=0,all_owners_retired=False,
        native_gradient_certified=False,raw_improvement_claimed=False,fit_authorized=False,
        original_native_fidelity_passed=False,NADJ_requalified=False,source_bindings={},
        input_bindings={},artifacts={},completed_calls=[])
    workspace=weights=thread_limits=None;refs=[]
    def tick():
        require(time.monotonic()-began<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8_000_000_000,
            'Whole120s/8GB; external watcher owns combined tree/no-swap/observation deadlines')
    def on_call(call,record,artifacts):
        require(call==len(result['completed_calls']) and record['call_graph_owners_retired'],
            'Ordered completed and retired graph, not whole-unit retirement')
        expected={f'call_{call:02d}_served.f32'}
        if call:expected.add(f'call_{call:02d}_gradients.f32')
        require(set(artifacts)==expected,'Only complete owned served/gradient bytes')
        for name,raw in artifacts.items():
            require(type(raw) is bytes and len(raw)==(180800 if 'gradients' in name else 65536),
                'Complete fixed eleven-P or all16 output export')
            with (output/name).open('xb') as stream:stream.write(raw)
            result['artifacts'][str(output/name)]=sha(output/name)
        path=output/f'call_{call:02d}.json';write(path,record)
        result['artifacts'][str(path)]=sha(path)
        result['completed_calls'].append(dict(call=call,arm=record['arm'],elapsed_s=record['model_elapsed_s'],
            call_graph_owners_retired=True))
        result['full_model_calls']+=1;result['backwards']+=int(record['backward'])
        print(json.dumps(dict(event='call_retired',call=call,arm=record['arm'],elapsed_s=record['model_elapsed_s'])),flush=True)
        tick()
    try:
        pins=json.loads(Path(args.pins).read_text())
        require(set(pins)==NAMES and all(sha(HERE/k)==v for k,v in pins.items()),'Four exact reviewed direct owners')
        require(Path(sys.executable).resolve()==Path('/Users/matt/anaconda3/bin/python').resolve()
            and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in LIMITS)
            and sys.flags.optimize==0 and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
            and not os.environ.get('FLOPSCOPE_SERVER_URL'),'Fixed CPU interpreter/nice10/BLAS1/assertions/noRPC')
        spec=importlib.util.spec_from_file_location('anchored_execution_qualification',HERE/'qualification.py')
        qualifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(qualifier)
        prepared=qualifier.prepare(pins['qualification.py'],pins['QUALIFICATION_PROTOCOL.md'],tick)
        result['source_bindings']=dict(prepared['source_bindings'])
        result['source_bindings'].update({str(HERE/k):v for k,v in pins.items()})
        result['source_bindings'][str(SUPERVISOR_OWNER)]=SUPERVISOR_SHA
        result['input_bindings']=dict(prepared['input_bindings'])
        result['cost_certificate']=prepared['cost_certificate']
        require(all(sha(p)==h for p,h in result['source_bindings'].items()),'Complete direct/inherited source admission')
        model=prepared['model'];bridge,bp=model.native_bridge,model.native_prepared
        result['backend_selection']=bridge.select_backend(bp,tick)
        import numpy as np
        import torch
        from threadpoolctl import threadpool_info,threadpool_limits
        thread_limits=threadpool_limits(limits=1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        attestation=bridge.attest(bp,np,torch,threadpool_info,tick)
        workspace=bridge.build_workspace(bp,np,attestation)
        refs.append(weakref.ref(workspace))
        def remember(value):
            if isinstance(value,np.ndarray):refs.append(weakref.ref(value))
            elif isinstance(value,dict):
                for x in value.values():remember(x)
            elif isinstance(value,(tuple,list)):
                for x in value:remember(x)
        remember(workspace.env['_SCRATCH'])
        require(workspace.env['_SCRATCH'][0].nbytes==1_168_900_096,'Fixed explicitly owned native workspace')
        weights=np.load(prepared['weights_path'],allow_pickle=False);remember(weights)
        require(weights.shape==(16,1024,1024) and weights.dtype==np.float32
            and weights.flags.owndata and np.isfinite(weights).all(),'Owned canonical finite physical weights')
        weights_sha=hashlib.sha256(weights.tobytes()).hexdigest()
        tick()
        answer,artifacts=qualifier.evaluate(prepared,np,torch,weights,workspace,attestation,on_call,tick)
        require(answer['passed'] and answer['completed'] and answer['full_model_calls']==3
            and answer['backwards']==2 and answer['all_helper_numerical_owners_retired']
            and answer['truth_accessed'] is False and answer['parameter_updates']==0,
            'Exact target-free three-call same-surrogate execution contract')
        require(len(artifacts)==5 and all(result['artifacts'][str(output/name)]==hashlib.sha256(raw).hexdigest()
            for name,raw in artifacts.items()),'Only already-published complete detached byte artifacts')
        require(hashlib.sha256(weights.tobytes()).hexdigest()==weights_sha,'Physical weight owner unchanged')
        remember(workspace.env.get('_PACKED_FRAME_CACHE',{}))
        workspace.close();del workspace,weights;workspace=weights=None;gc.collect()
        require(all(ref() is None for ref in refs),'All caller workspace/scratch/weight owners retired')
        final_attestation=bridge.attest(bp,np,torch,threadpool_info,tick)
        require(all(final_attestation['source_bindings'].get(p)==h
            for p,h in attestation['source_bindings'].items()),'Same actual loaded numerical origins')
        for p,h in final_attestation['source_bindings'].items():
            require(p not in result['source_bindings'] or result['source_bindings'][p]==h,'No backend owner conflict')
            result['source_bindings'][p]=h
        for group in ('source_bindings','input_bindings'):
            for p,h in result[group].items():
                require(sha(p)==h,'Every actual source/input owner unchanged');tick()
        path=output/'diagnostics.json';write(path,answer);result['artifacts'][str(path)]=sha(path)
        result.update(passed=True,completed=True,all_owners_retired=True,parameters_unchanged=True,
            backend_attestation=final_attestation,execution_integrity_qualified=True,
            native_score_measured=False,complete_source_upper=prepared['cost_certificate']['complete_source_upper'],
            actual_flopscope_bill_claimed=False,qualification_scope='same_surrogate_execution_integrity_only')
    except BaseException:
        result['error']=traceback.format_exc()
    finally:
        if workspace is not None:
            try:workspace.close()
            except BaseException:result['workspace_close_error']=traceback.format_exc()
        workspace=weights=None;gc.collect()
        if thread_limits is not None:thread_limits.restore_original_limits()
    result.update(elapsed_s=time.monotonic()-began,
        maximum_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s']>=120 or result['maximum_self_rss_bytes']>=8_000_000_000:
        result.update(passed=False,completed=False,resource_error=True)
    write(output/'result.json',result);rh=sha(output/'result.json')
    write(output/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh))
    write(output/'finalization.json',dict(passed=result['passed'],completed=result['completed'],
        result_sha256=rh,completion_sha256=sha(output/'completion.json'),elapsed_s=time.monotonic()-began))
    signal.alarm(0)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','full_model_calls','backwards',
        'elapsed_s','maximum_self_rss_bytes','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
