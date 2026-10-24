"""One target-blind unchanged current-c2 cold35, metered L8/L9 restart packet.

SOURCE DRAFT: explicit --run only. No target reader, local intervention, fit,
native restart, bank decision, or automatic later stage is implemented here.
"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback
import types
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
SHARED = NEW/'current_comp_shared_old384_v1'
GAIN = NEW/'current_comp_shared_gain_slice_v1'
OFFPY = P2/'data_gate/.venv/bin/python'
ROW = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/0035'
OUTPUT = HERE/'capture_v1'
CAP, RSS, BASE_F, INIT = 180., 8_000_000_000, 403400296797, 1944970298
SCHEMA = 'current_comp_joint_cell_capture_v1'
KEYS = ('source','adapter','protocol','watcher','watch-spec','design')
PINS = {
    SHARED/'probe.py':'6193396f83150e7e95f3c16ad5095eeb8d4c62da17d69d5bd7dcb08fad9a743b',
    SHARED/'backend_probe.py':'36d9068c8dead7f32b53da9c5c5382f9004a8a45c0c47c29d7e1f4b2435fdcc8',
    SHARED/'check4_v2.py':'8d09d9b1f443e229b1641d277312b52c204392648b25445ffab63b82dba16058',
    NEW/'birth_direction_fit_v1/native_v2.py':'8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    NEW/'current_comp_aligned_push_d6_v1/native.py':'751bc155506b2825a8c0d27553e8ced8d7db66d68434247bb158c22fad3c3ad6'}
CONTROL = {
    'native_panel_v1':('9563919f22a938ec084751444cf1c4e643e18fd4aba1cf8b87c3526f2d3f6608',
        'fe75d8c7943c14a58a66025f96ecd4e97b8f89d44d12759a0d1ff2233f8bbf4c',
        '3578b8eb6a9c9a1c183d9c87c3dd32c3a9fa322bfa70209ecc3a6c0f4aeeb36e'),
    'native_panel_watch_v1':('6fe4acd894a5dd13f0bdb558ef48c522622917413b400a0b1ca43850236474fa',
        '6d58d72298d7cda2049b995b3faf4c0bd090394a03b3b61d3cda35ca6a81c040',
        'e9bf477c760ca54805e6610407d0b5d2133035679ef989cb5e35ca40a71ea088')}
FIXTURE = {ROW/'row_manifest.json':'5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
    ROW/'weights.npy':'14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'}
OPEN_SITE=NEW/'openblas_runtime_env_v1/lib/python3.11/site-packages/flopscope'
PINS.update({OPEN_SITE/'data/default_weights.json':'f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a',
    OPEN_SITE/'_weights.py':'22591dd4dd2d54b4134da82f67eb9c299a7e4452f7d815f2f54d2b96fb9633a2'})


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path,tick=None):
    path=Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and
        not (path.suffix in ('.npy','.npz') and 'truth' in str(path).lower()), 'No target path access')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
            if tick is not None: tick()
    return h.hexdigest()


def read(path):
    path=Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2,'Bounded actual JSON')
    return json.loads(path.read_text())


def write(path,value):
    def safe(x):
        if isinstance(x,float) and not math.isfinite(x): return dict(nonfinite_float=repr(x))
        if isinstance(x,dict): return {k:safe(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [safe(v) for v in x]
        return x
    with Path(path).open('x') as f:
        json.dump(safe(value),f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')


def load(path,name,digest):
    require(sha(path)==digest,'Exact source before inert import: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def bind(values,tick=None):
    for p,d in values.items(): require(sha(p,tick)==d,'Fresh immutable source/input/artifact: '+str(p))


def admission(args,tick=None):
    admission_began=time.monotonic(); outer_tick=tick
    def tick():
        require(time.monotonic()-admission_began<60 and
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1_000_000_000,
            'Target-free metadata admission60s/1GB before any numerical import')
        if outer_tick is not None: outer_tick()
    tick()
    names=('capture.py','adapter.py','CAPTURE_PROTOCOL.md','watch_capture.py','CAPTURE_WATCH_SPEC.md','DESIGN.md')
    own={str(HERE/name):getattr(args,key.replace('-','_')+'_sha256') for name,key in zip(names,KEYS)}
    require(all(re.fullmatch('[0-9a-f]{64}',x or '') for x in own.values()),'All six reviewed owners explicit')
    own.update({str(p):d for p,d in PINS.items()}); bind(own,tick)
    require(not os.environ.get('FLOPSCOPE_DISABLE_WEIGHTS') and not os.environ.get('FLOPSCOPE_WEIGHTS_FILE'),
            'No tariff override or disabling')
    tariffs=read(OPEN_SITE/'data/default_weights.json')
    require(all(tariffs['weights'][k]==1 for k in ('copy','isfinite','all','multiply','divide'))
        and all(tariffs['dtype_rates'][k]==v for k,v in
            dict(bool=1,float32=1,float64=2,int32=1,int64=2).items()), 'Exact source tariff basis')
    inputs={str(p):d for p,d in FIXTURE.items()}; bind(inputs,tick)
    records={}
    for folder,ds in CONTROL.items():
        pp=[GAIN/folder/n for n in ('result.json','completion.json','finalization.json')]
        require(tuple(sha(p,tick) for p in pp)==ds,'Actual current-c2 direct three-seal boundary')
        r,e,f=map(read,pp)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256']==f['result_sha256']==ds[0] and f['completion_sha256']==ds[1],
            'Current-c2 genuinely completed, not an old false-watch repair')
        inputs.update({str(p):d for p,d in zip(pp,ds)}); records[folder]=r
    r,w=records['native_panel_v1'],records['native_panel_watch_v1']
    require(r['schema']=='current_comp_shared_gain_slice_native_panel_v1'
        and w['candidate_sha256']==r['candidate_sha256']
        and [w['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')]
            ==list(CONTROL['native_panel_v1']) and w['whole_owned_tree_retired'] and w['cleanup_passed']
        and w['maximum_combined_rss_bytes']<RSS and w['maximum_interval_s']<=1
        and w['maximum_swap_growth_bytes']==0,'Positive current backend/resource control')
    c=r['trials'][0]
    require(c['rank']==35 and c['call']==0 and c['includes_initialization'] and c['flops']==BASE_F
        and c['operation_count']==63927 and c['output_sha256']==r['trials'][14]['output_sha256'],
        'Fixed unchanged cold35 and authentic own repeat, not best-value selection')
    paths=[c['output'],str(GAIN/'native_panel_v1/call_00_bill.json'),c['input_bill'],
        c['input_reset_receipt'],c['lifecycle_receipt'],str(GAIN/'native_panel_v1/call_00_diagnostics.json'),
        str(GAIN/'native_panel_v1/call_00_complete.json')]
    for p in paths:
        require(sha(p,tick)==r['artifacts'][p],'Every selected current35 artifact sealed')
        inputs[p]=r['artifacts'][p]
    require(read(c['input_bill'])['weight_file_sha256']==FIXTURE[ROW/'weights.npy'], 'Identical physical fixture')
    a=load(HERE/'adapter.py','joint_cell_inert_adapter',args.adapter_sha256)
    require(a.COPY_CAP==480*1024**2 and 3*a.COPY_CAP+64*1024**2<a.DIAGNOSTIC_CAP,
            'Before model: copy+finite+all <=3F/byte plus64MiF margin for two physical Q31 planes')
    require(r['candidate_sha256']==a.CANDIDATE_SHA and sha(a.CANDIDATE,tick)==a.CANDIDATE_SHA,
            'One current-c2 parent, not a new numerical candidate')
    own[str(a.CANDIDATE)]=a.CANDIDATE_SHA
    h=load(SHARED/'probe.py','joint_cell_paid_owner',PINS[SHARED/'probe.py'])
    backend=load(SHARED/'backend_probe.py','joint_cell_backend_owner',PINS[SHARED/'backend_probe.py'])
    own.update({str(p):d for p,d in backend._BP_PINS.items()}); bind(own,tick)
    owner=h.load(h.OBSERVER,'joint_cell_actual_observer_owner',h.OBSERVER_SHA)
    own.update({str(p):d for p,d in owner.OWNERS.items()}); bind(own,tick)
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),
            'Metadata admission imported no numerical package')
    return dict(source_bindings=own,input_bindings=inputs,control=c,
        diagnostic=read(paths[-2]),control_seals=CONTROL),a,h,backend,owner


def prepare(h,a):
    loop,proof=h.paid_loop_source(); original=loop; spans=[]
    edits=[('enumerate(tuple(ranks)+(ranks[0],))','enumerate(tuple(ranks))'),
        ('            observer.begin(); observer.ctx=ctx',
         '            observer.begin(); observer.ctx=ctx; sink.ctx=ctx'),
        ('                observer.ctx=None','                observer.ctx=None; sink.ctx=None'),
        ('        if call==0: first_bytes=values.tobytes()\n',''),
        ("        if call==len(ranks): assert values.tobytes()==first_bytes,'Repeat identity'\n",'')]
    for old,new in edits:
        require(loop.count(old)==1,'One-call lifecycle splice: '+old)
        p=loop.index(old); loop=loop[:p]+new+loop[p+len(old):]; spans.append((p,old,new))
    inverse=loop
    for p,old,new in reversed(spans):
        require(inverse[p:p+len(new)]==new,'Exact positional one-call inverse')
        inverse=inverse[:p]+old+inverse[p+len(new):]
    require(inverse==original,'Complete original paid-loop inverse')
    return loop,dict(owner=proof,one_call_inverse=True,derived_sha256=hashlib.sha256(loop.encode()).hexdigest(),
        no_new_repeat_claim=True,actual_setup_predict_materialize_bill_reset_release_preserved=True)


def filtered_ledger(ctx,spans):
    """Exact original encoding, skipping only recorded callback spans; no log edit."""
    h=hashlib.sha256(); count=flops=0; end=0
    for span in spans:
        require(end<=span['start']<span['stop']<=len(ctx.op_log),'Nonoverlapping real callback spans')
        require(sum(int(x.flop_cost) for x in ctx.op_log[span['start']:span['stop']])==span['flops'],
                'Every removed diagnostic tariff comes from original actual op log')
        end=span['stop']
    def shape(v):
        if isinstance(v,(tuple,list)): return [shape(x) for x in v]
        if v is None or isinstance(v,str): return v
        return int(v)
    for i,op in enumerate(ctx.op_log):
        if any(s['start']<=i<s['stop'] for s in spans): continue
        encoded=json.dumps([str(op.op_name),op.subscripts,shape(op.shapes),int(op.flop_cost),op.resolved_dtype],
            separators=(',',':'),allow_nan=False).encode()+b'\n'
        h.update(encoded); count+=1; flops+=int(op.flop_cost)
    return dict(operation_count=count,flops=flops,all_records_sha256=h.hexdigest(),
        diagnostic_subtraction_for_control_check_only=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    for key in KEYS: parser.add_argument('--'+key+'-sha256')
    args=parser.parse_args()
    if not args.run: parser.print_help(); return 0
    require(not OUTPUT.exists() and not HERE.is_symlink(),'Fresh output; never append failure seals to an old stage')
    began=time.monotonic(); OUTPUT.mkdir(); args.out=OUTPUT
    result=dict(schema=SCHEMA,passed=False,completed=False,source_bindings={},input_bindings={},artifacts={},
        trials=[],input_preparation=[],diagnostic_records=[],ranks=[35],full_model_calls=0,retired_call_count=0,
        truth_accessed=False,score_measured=False,parameter_updates=0,automatic_next_stage=False,
        repeat_acquired=False,repeat_identity_claimed=False,consumer_qualified=False)
    def tick():
        require(time.monotonic()-began<CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<RSS,
                'Whole180s/native8GB; independent watcher owns full tree')
    try:
        context,a,h,backend,owner=admission(args,tick)
        result.update({k:context[k] for k in ('source_bindings','input_bindings','control_seals')})
        text,proof=a.derive_capture(); loop,loop_proof=prepare(h,a)
        result.update(candidate_sha256=a.CANDIDATE_SHA,capture_source_proof=proof,paid_loop_proof=loop_proof)
        Parent,diagnostics,observer_proof=h.observers(owner)
        legacy=h.load(h.LEGACY,'joint_cell_original_paid_lifecycle',h.LEGACY_SHA)
        result['source_bindings'].update({str(p):d for p,d in legacy.PINS.items()})
        ledger_text=(SHARED/'check4_v2.py').read_text(); lo,hi=a.function_span(ledger_text,'_v2_ledger_metrics')
        ledger_ns=dict(time=time,hashlib=hashlib,json=json,require=require)
        exec(compile(ledger_text.encode()[lo:hi].decode(),'<exact-closed-ledger-owner>','exec'),ledger_ns)
        ledger_metrics=ledger_ns['_v2_ledger_metrics'](legacy.metrics)
        bind(result['source_bindings'],tick); bind(result['input_bindings'],tick)
        backend.backend_select(); h.environment()
        require(not {'numpy','torch','flopscope'} & set(sys.modules),'Admission before numerical imports')
        write(OUTPUT/'started.json',result)
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        module=types.ModuleType('joint_cell_capture_subject'); module.__file__='<fixed-c2-capture>'
        sink=a.Sink(fnp); module._jca_emit=sink
        exec(compile(text,module.__file__,'exec'),vars(module)); observer=Parent(module)
        result['observer_proof']=observer_proof
        def metrics(ctx,started):
            value=ledger_metrics(ctx,started)
            if sink.spans and int(ctx.flops_used)>BASE_F:
                value['unchanged_model_subsequence']=filtered_ledger(ctx,sink.spans)
            return value
        def finish_call(np,observer,call,rank,result,folder,trial):
            packet,startup,dispatches=observer.finish(); domain=diagnostics(np,packet)
            old=context['diagnostic']
            require(domain==old['domains'] and startup==old['startup'] and dispatches==old['dispatches'],
                    'All16/all47/current dispatch/startup exact, not prefix-only')
            require(startup['observed_startup_region_flops']==INIT,'Actual cold startup fully owned')
            require(trial['output_sha256']==context['control']['output_sha256'] and
                trial['output_bytes_sha256']==context['control']['output_bytes_sha256'], 'All16 output bytes exact')
            sub=trial['unchanged_model_subsequence']; oldledger=context['control']['operation_ledger']
            require(sub['flops']==BASE_F and sub['operation_count']==63927 and
                sub['all_records_sha256']==oldledger['all_records_sha256'], 'Entire original typed operation sequence exact')
            manifest=sink.finish()
            require(trial['flops']==BASE_F+manifest['diagnostic_flops']<=405500000000,
                    'Actual complete own bill includes every diagnostic operation')
            path=folder/'restart_packet.bin'
            with path.open('xb') as stream:
                for raw in sink.blocks: stream.write(raw)
            sink.blocks.clear()
            require(path.stat().st_size==manifest['bytes']<=a.PACKET_CAP,'Complete detached packet byte cap')
            manifest.update(packet_file=str(path),packet_sha256=sha(path),saved_after_context_release=True)
            mp=folder/'restart_packet.json'; write(mp,manifest)
            dp=folder/'call_00_diagnostics.json'
            write(dp,dict(call=0,rank=35,domains=domain,startup=startup,dispatches=dispatches,
                          metered_diagnostic_flops=manifest['diagnostic_flops']))
            for p in (path,mp,dp): result['artifacts'][str(p)]=sha(p)
            result['diagnostic_records'].append(dict(path=str(dp),sha256=sha(dp)))
            result.update(full_model_calls=1,baseline_output_byte_exact=True,all16_domains_qualified=True,
                all47_selectors_qualified=True,all_original_operations_exact=True,own_complete_bills=True,
                initialization_flops=INIT,cold_minus_repeat_measured=False,packet_bytes=manifest['bytes'],
                diagnostic_flops=manifest['diagnostic_flops'],all_capture_arrays_released=True)
            del packet; tick()
        def retire_call(call,rank,result,folder,trial):
            life=read(trial['lifecycle_receipt'])
            require(call==0 and rank==35 and life['context_released'] and life['complete'] and
                life['closed_context_bill_unchanged_by_reset'] and observer.ctx is None and sink.ctx is None,
                'Actual complete prediction/input/reset/released context')
            p=folder/'call_00_complete.json'
            write(p,dict(passed=True,completed=True,call=0,rank=35,context_retired=True,
                bill_sha256=sha(folder/'call_00_bill.json'),lifecycle_sha256=sha(trial['lifecycle_receipt']),
                diagnostic_sha256=result['diagnostic_records'][0]['sha256'],output_sha256=trial['output_sha256']))
            result['artifacts'][str(p)]=sha(p); result['retired_call_count']=1
            print(json.dumps(dict(event='retired',call=0,rank=35,complete_sha256=sha(p))),flush=True)
        row=read(ROW/'row_manifest.json')
        env=dict(vars(legacy),np=np,flops=flops,fnp=fnp,threadpool_limits=threadpool_limits,
            threadpool_info=backend.backend_threadpool_info(threadpool_info,result,tick),MLP=MLP,
            derive_estimator_seed=derive_estimator_seed,materialise_predictions=materialise_predictions,args=args,
            inputs=result['input_bindings'],rows={35:dict(weights=str(ROW/'weights.npy'),mlp_seed=int(row['mlp_seed']))},
            ranks=(35,),result=result,module=module,estimator=None,observer=observer,sink=sink,
            observed_predict=owner.base().observed_predict,finish_call=finish_call,retire_call=retire_call,
            metrics=metrics,sha=sha,write=write)
        exec(compile(loop,'<exact-single-paid-cold35-loop>','exec'),env)
        require(result['full_model_calls']==result['retired_call_count']==1,'Exactly one full model, no hidden repeat')
        module_ref=weakref.ref(module); observer_ref=weakref.ref(observer); sink_ref=weakref.ref(sink)
        # Callback closures are severed before checking private subject retirement.
        del env,observer,module,sink; gc.collect()
        require(module_ref() is observer_ref() is sink_ref() is None,'Subject/callback/observer owners retired')
        result.update(all_parameter_and_estimator_owners_retired=True)
        for group in ('source_bindings','input_bindings','artifacts'): bind(result[group],tick)
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s']>=CAP or result['peak_rss_bytes']>=RSS: result.update(passed=False,completed=False)
    write(OUTPUT/'result.json',result); rh=sha(OUTPUT/'result.json'); ok=bool(result['passed'] and time.monotonic()-began<CAP)
    write(OUTPUT/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch=sha(OUTPUT/'completion.json')
    elapsed=time.monotonic()-began; ok=bool(ok and elapsed<CAP)
    write(OUTPUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
    return 0 if ok and time.monotonic()-began<CAP else 1


if __name__=='__main__': raise SystemExit(main())
