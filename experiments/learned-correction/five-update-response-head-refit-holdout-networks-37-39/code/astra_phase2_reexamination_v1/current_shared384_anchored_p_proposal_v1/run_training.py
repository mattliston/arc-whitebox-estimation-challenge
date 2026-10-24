"""One fixed anchored TRAIN20 dose; native acceptance is a separate experiment."""
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
NAMES={'run_training.py','watch_training.py','train_engine.py','proposal_admission.py','TRAIN_PROTOCOL.md'}
SUPERVISOR_OWNER=HERE.parent/'current_native_trajectory_adjoint_v1/supervise.py'
SUPERVISOR_SHA='648e5227795a8d2e5a91a1a421e0e54ffea1f76d6d6d65b28335178c8772025d'
LIMITS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
CAP=1500.;RSS=8_000_000_000


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def load(path,expected,name):
    require(sha(path)==expected,'Exact reviewed source before inert import')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true');ap.add_argument('--pins')
    args=ap.parse_args()
    if not args.run:ap.print_help();return 0
    began=time.monotonic();output=HERE/'training_v1'
    require(not output.exists(),'One fresh training dose, no resume or overwrite');output.mkdir()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Whole TRAIN20 stage1500s')))
    signal.alarm(1500)
    result=dict(passed=False,completed=False,fits=0,full_model_calls=0,backwards=0,optimizer_steps=0,
        truth_accessed=False,source_bindings={},input_bindings={},truth_bindings={},artifacts={},
        calls=[],updates=[],payloads=[],all_owners_retired=False,native_score_measured=False,
        native_gradient_certified=False,release_ready=False,automatic_next_stage=False)
    workspace=thread_limits=None;refs=[];prepared=None
    def tick():
        require(time.monotonic()-began<CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<RSS,
            'Whole1500s/self8GB; independent watcher additionally owns tree/swap/progress deadlines')
    def artifact(name,raw):
        require(Path(name).name==name and type(raw) is bytes,'Exclusive detached byte artifact')
        path=output/name
        with path.open('xb') as stream:stream.write(raw)
        result['artifacts'][str(path)]=sha(path)
        return path
    def on_call(index,record,artifacts):
        require(prepared is not None and index==len(result['calls']) and 0<=index<60,
            'Exactly sixty ordered calls')
        stage=('baseline','training','terminal')[index//20];rank=prepared['train'][index%20]
        require(record['rank']==rank and record['stage']==stage
            and record['call_graph_owners_retired'] and record['input_owners_retired'],
            'Only completed retired graph/card/weight calls may mark progress')
        expected={f'call_{index:02d}_served.f32'}
        if index>=20:expected.add(f'call_{index:02d}_anchored.f32')
        require(set(artifacts)==expected and all(type(raw) is bytes and len(raw)==65536
            for raw in artifacts.values()),'Fixed complete all16 FP32 call artifacts')
        for name,raw in artifacts.items():artifact(name,raw)
        path=output/f'call_{index:03d}.json';write(path,record)
        digest=sha(path);result['artifacts'][str(path)]=digest
        result['calls'].append(dict(index=index,stage=stage,rank=rank,path=str(path),sha256=digest,
            model_elapsed_s=record['model_elapsed_s'],call_graph_owners_retired=True,input_owners_retired=True))
        result['full_model_calls']+=1;result['backwards']+=int(stage=='training')
        result['truth_accessed']=bool(index>=20)
        marker=dict(schema='anchored_train_call_retired_v1',index=index,rank=rank,stage=stage,
            call_graph_owners_retired=True,input_owners_retired=True,
            shared_state_owners_may_persist=True,record_path=str(path),record_sha256=digest)
        staged=output/f'call_{index:03d}.retired.staged.json';write(staged,marker)
        published=output/f'call_{index:03d}.retired.json'
        os.link(staged,published)  # Atomic publication after close; retain both links, delete nothing.
        result['artifacts'][str(staged)]=sha(staged);result['artifacts'][str(published)]=sha(published)
        print(json.dumps(dict(event='call_retired',index=index,stage=stage,rank=rank,
            model_elapsed_s=record['model_elapsed_s'])),flush=True)
        tick()
    def on_update(index,record):
        require(index==len(result['updates']) and 0<=index<5 and record['optimizer_step']==index+1,
            'Exactly five fixed ordered Adam updates, no checkpoint selection')
        path=output/f'update_{index:02d}.json';write(path,record)
        result['artifacts'][str(path)]=sha(path);result['updates'].append(dict(index=index,path=str(path),sha256=sha(path)))
        result['optimizer_steps']+=1;result['fits']=1;tick()
    def on_payload(record,raw):
        require(len(result['payloads'])==0 and result['optimizer_steps']==5 and result['full_model_calls']==40,
            'Only the terminal five-update payload, before fixed terminal forwards')
        path=artifact('terminal_payload.npz',raw)
        require(record['sha256']==sha(path),'Complete sole payload byte identity')
        receipt=output/'terminal_payload.json';write(receipt,record)
        result['artifacts'][str(receipt)]=sha(receipt)
        result['payloads'].append(dict(path=str(path),sha256=sha(path),record=str(receipt)))
        tick()
    try:
        pins=json.loads(Path(args.pins).read_text())
        require(set(pins)==NAMES and all(sha(HERE/k)==v for k,v in pins.items()),'Five exact reviewed TRAIN20 owners')
        require(Path(sys.executable).resolve()==Path('/Users/matt/anaconda3/bin/python').resolve()
            and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in LIMITS)
            and sys.flags.optimize==0 and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
            and not os.environ.get('FLOPSCOPE_SERVER_URL'),'Fixed CPU/nice10/BLAS1/assertions/no external endpoint')
        admission=load(HERE/'proposal_admission.py',pins['proposal_admission.py'],'anchored_TRAIN20_admission')
        engine=load(HERE/'train_engine.py',pins['train_engine.py'],'anchored_TRAIN20_engine')
        prepared=admission.admission(pins['proposal_admission.py'],pins['TRAIN_PROTOCOL.md'],tick)
        require(prepared['whole_seconds']==CAP and prepared['rss_bytes']==RSS
            and prepared['cost_certificate']['complete_source_upper']==134_939_477_450_240,
            'Fixed complete prospective dose and max-geometry price')
        result['source_bindings']=dict(prepared['source_bindings'])
        result['source_bindings'].update({str(HERE/k):v for k,v in pins.items()})
        result['source_bindings'][str(SUPERVISOR_OWNER)]=SUPERVISOR_SHA
        result['input_bindings']=dict(prepared['input_bindings'])
        result['truth_bindings']=dict(prepared['truth_bindings'])
        result['cost_certificate']=prepared['cost_certificate']
        require(all(sha(p)==h for p,h in result['source_bindings'].items()),'Actual complete source admission')
        model=prepared['model'];bridge,bp=model.native_bridge,model.native_prepared
        result['backend_selection']=bridge.select_backend(bp,tick)
        import numpy as np
        import torch
        from threadpoolctl import threadpool_info,threadpool_limits
        thread_limits=threadpool_limits(limits=1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        attestation=bridge.attest(bp,np,torch,threadpool_info,tick)
        workspace=bridge.build_workspace(bp,np,attestation);refs.append(weakref.ref(workspace))
        def remember(value):
            if isinstance(value,np.ndarray):refs.append(weakref.ref(value))
            elif isinstance(value,dict):
                for x in value.values():remember(x)
            elif isinstance(value,(tuple,list)):
                for x in value:remember(x)
        remember(workspace.env['_SCRATCH'])
        require(workspace.env['_SCRATCH'][0].nbytes==1_168_900_096,'Same explicit qualified native workspace')
        answer,artifacts=engine.evaluate(prepared,np,torch,workspace,attestation,on_call,on_update,on_payload,tick)
        require(answer['passed'] and answer['completed'] and answer['all_engine_numerical_owners_retired']
            and result['full_model_calls']==60 and result['backwards']==20 and result['optimizer_steps']==5
            and len(result['payloads'])==1,'One complete sole-payload fixed-dose training stage')
        for name,raw in artifacts.items():
            path=output/name
            if path.exists():
                require(result['artifacts'].get(str(path))==hashlib.sha256(raw).hexdigest(),'Already published identical artifact')
            else:artifact(name,raw)
        remember(workspace.env.get('_PACKED_FRAME_CACHE',{}));workspace.close();workspace=None;gc.collect()
        require(all(ref() is None for ref in refs),'All root workspace/scratch owners retired')
        final_attestation=bridge.attest(bp,np,torch,threadpool_info,tick)
        require(all(final_attestation['source_bindings'].get(p)==h for p,h in attestation['source_bindings'].items()),
            'Initial and final actual numerical owners agree')
        for p,h in final_attestation['source_bindings'].items():
            require(p not in result['source_bindings'] or result['source_bindings'][p]==h,'No imported-owner conflict')
            result['source_bindings'][p]=h
        for group in ('source_bindings','input_bindings','truth_bindings'):
            for p,h in result[group].items():
                require(sha(p)==h,'Every actual source/input/TRAIN truth owner unchanged');tick()
        path=output/'diagnostics.json';write(path,answer);result['artifacts'][str(path)]=sha(path)
        result.update(passed=True,completed=True,fits=1,truth_accessed=True,all_owners_retired=True,
            backend_attestation=final_attestation,training_dose_completed=True,
            complete_source_upper=prepared['cost_certificate']['complete_source_upper'],
            actual_flopscope_bill_claimed=False,qualification_scope='approximate_proposal_only_native_acceptance_pending')
    except BaseException:
        result['error']=traceback.format_exc()
        if result['full_model_calls']==20 and result['truth_accessed'] is False:
            result['truth_accessed']=None
            result['truth_access_status']='Authorized target reads may have occurred after baseline; no completed training call observed.'
    finally:
        if workspace is not None:
            try:workspace.close()
            except BaseException:result['workspace_close_error']=traceback.format_exc()
        workspace=None;gc.collect()
        if thread_limits is not None:thread_limits.restore_original_limits()
    result.update(elapsed_s=time.monotonic()-began,maximum_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s']>=CAP or result['maximum_self_rss_bytes']>=RSS:
        result.update(passed=False,completed=False,resource_error=True)
    write(output/'result.json',result);rh=sha(output/'result.json')
    write(output/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh))
    write(output/'finalization.json',dict(passed=result['passed'],completed=result['completed'],
        result_sha256=rh,completion_sha256=sha(output/'completion.json'),elapsed_s=time.monotonic()-began))
    signal.alarm(0)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','full_model_calls','backwards',
        'optimizer_steps','elapsed_s','maximum_self_rss_bytes','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
