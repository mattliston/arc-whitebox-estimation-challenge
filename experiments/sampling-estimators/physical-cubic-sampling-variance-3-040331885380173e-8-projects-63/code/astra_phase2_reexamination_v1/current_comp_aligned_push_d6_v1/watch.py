"""One target-free aligned-PUSH build and two-context runtime falsifier."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import struct
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent,HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
TRANSPORT = NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
CD = NEW/'current_comp_partial_d6_v1'
CD_SHA = 'ca33a2a4600e7adb10071f6f904936eefe13915278513a8e6291fb5d3e6b562d'
CONTROL = CD/'production_watch_v1'
CONTROL_PINS = {
    CONTROL/'result.json':'422407e0965f55451f2fc1ef3f8bc3aefdfe18c270c99f825b6054369aa054f2',
    CONTROL/'completion.json':'164abb69d2d9efa61493760be0248d63bfc0d8eaafa2b6520ca75f943b9d7058',
    CONTROL/'finalization.json':'ed25014230dcd521623fbc13627a9422b7de89949ada88da8d091ee29b02fcd2',
    CD/'PUSH_PRICE_20260910.md':'b3395f73b83ff6a69b6109432779664754890cc0e3d9312d99d1691227bfbd30',
}
ROW = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/0035'
FIXTURE_PINS = {
    ROW/'weights.npy':'14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
    ROW/'row_manifest.json':'5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
    ROW.parent.parent.parent/'manifest.json':'9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
}
ORIGINALS = {
    NEW/'run_remote_series_v1.py':'00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1',
    NEW/'check_remote_series_v1.py':'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55',
    CD/'build_v1/estimator.py':CD_SHA,
}
CANDIDATE = 'current_comp_aligned_push_d6_v1/build_v1/estimator.py'
TAG = 'current_comp_aligned_push_d6_runtime_v1'
RANKS = (35,35)
INIT, BUDGET, CAP, RSS = 1944970298,2**41,300.,8_000_000_000
BLAS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def load(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transport():
    require(hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == TRANSPORT_SHA,'Qualified transport')
    h = load(TRANSPORT,'aligned_push_transport')
    mon,classifier = h.transport()
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'Stdlib-only supervisor')
    return h,mon,classifier


def check_bill(row,index):
    require(type(index) is int and 0 <= index < 2 and row['index'] == index
        and row['rank'] == 35 and row['output_shape'] == [16,1024]
        and row['output_dtype'] == 'float32','Two actual35 contexts and all16 output shape')
    f,t = row['flops'],row['timing']
    require(type(f) is int and 0 < f <= BUDGET and row['util'] == f/BUDGET,'Own full actual bill')
    require(isinstance(t,dict) and t and all(type(v) in (int,float) and math.isfinite(v) and v >= 0
        for v in t.values()) and 0 < t['wall_time_s'] < 60 and t['residual_wall_time_s'] < .4,
        'Original per-call wall<60s/residual<.4s guards')
    require(re.fullmatch('[0-9a-f]{64}',row['output_sha256'] or ''),'Actual output hash')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--numerical-slot-clear',action='store_true')
    keys = ('source','build','client','runner','adapters','protocol')
    for key in keys:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    began = previous = last_progress = time.monotonic()
    require(args.numerical_slot_clear and os.getpriority(os.PRIO_PROCESS,0) >= 10
        and sys.flags.optimize == 0 and sys.dont_write_bytecode
        and Path(sys.prefix).resolve() == OFFPY.parent.parent.resolve()
        and all(os.environ.get(k) == '1' for k in BLAS),'Exclusive official nice10/BLAS1/bytecodeoff/assertions')
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in keys),'Reviewed source hashes')
    h,mon,classifier = transport()
    own = {Path(__file__).resolve():args.source_sha256,HERE/'build.py':args.build_sha256,
        HERE/'client.py':args.client_sha256,HERE/'runner.py':args.runner_sha256,
        HERE/'ADAPTERS.md':args.adapters_sha256,HERE/'DECISION.md':args.protocol_sha256,
        TRANSPORT:TRANSPORT_SHA,**h.MON_PINS,**ORIGINALS}
    inputs = {**h.MOCK_PINS,**CONTROL_PINS,**FIXTURE_PINS}
    out = HERE/'runtime_v1'
    series,usage,server_log = [NEW/(TAG+x) for x in ('.json','_resources.json','_server.log')]
    outputs = [NEW/f'{TAG}_{i}_r35.f32' for i in range(2)]
    require(not out.exists() and not (HERE/'build_v1').exists()
        and not any(p.exists() for p in (series,usage,server_log,*outputs)) and not HERE.is_symlink(),
        'Exclusive build/runtime stage, never overwrite or resume')
    out.mkdir()
    result = dict(schema='current_comp_aligned_push_d6_runtime_v1',passed=False,completed=False,
        source_bindings={str(p):v for p,v in own.items()},input_bindings={str(p):v for p,v in inputs.items()},
        artifacts={},candidate=CANDIDATE,tag=TAG,ranks=list(RANKS),
        numerical_imports=False,truth_arrays_accessed=False,raw_error_measured=False,
        adjusted_score_measured=False,production_five_qualified=False,uploaded=False,
        automatic_next_stage=False,whole_cap_s=CAP,progress_cap_s=120.,combined_rss_limit_bytes=RSS,
        full_model_calls=0,observed_records=[],completed_sample_intervals_s=[],
        maximum_combined_rss_bytes=0,maximum_interval_s=0.,maximum_swap_growth_bytes=0.,
        hashes_unchanged=False,sources_unchanged=False,whole_owned_tree_retired=False)
    child=log=reader=stream=None;swap0=None;samples=0;pending=''

    def deadline():
        now = time.monotonic()
        require(now-began < CAP and now-last_progress < 120.,'Whole300s/next-complete-context120s')

    def observe():
        nonlocal previous,last_progress,swap0,samples,pending
        absolute = min(previous+1.,began+CAP,last_progress+120.)
        snapshot = mon.collect(absolute)
        scope = mon.scope(snapshot['rows'],os.getpid(),None if child is None else child.pid)
        allowed = scope.pop('allowed')
        require(not classifier(snapshot['rows'],allowed),'No foreign numerical worker')
        for row in snapshot['rows']:
            if row['pid'] not in allowed:
                try: argv = shlex.split(row['command'])
                except ValueError: argv = row['command'].split()
                require('flopscope_server' not in argv,'No foreign numerical server')
        if swap0 is None:
            swap0 = snapshot['swap']['used_bytes']
        growth = snapshot['swap']['used_bytes']-swap0
        require(growth <= 0 and 0 < scope['combined_rss_bytes'] < RSS,'Zero added swap/full-owned-tree<8GB')
        if reader is not None:
            pending += reader.read()
            require(len(pending) < 2*1024**2,'Bounded original stdout tail')
            while '\n' in pending:
                line,pending = pending.split('\n',1)
                try: row = json.loads(line)
                except (ValueError,TypeError): continue
                if isinstance(row,dict) and 'index' in row and 'rank' in row:
                    deadline();index=len(result['observed_records']);check_bill(row,index)
                    require(outputs[index].stat().st_size == 16*1024*4
                        and h.sha(outputs[index]) == row['output_sha256'],'Complete saved output precedes progress')
                    result['observed_records'].append(row);result['full_model_calls']=index+1
                    last_progress=time.monotonic()
        require(snapshot['commands_reaped'] and not mon.PENDING,'Observation groups retired')
        deadline()
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,**scope,
            swap_bytes=snapshot['swap']['used_bytes'],swap_growth_bytes=growth,
            completed_calls=len(result['observed_records']),transport=snapshot['command_records'],
            uncached_single_process_table=True),allow_nan=False)+'\n')
        stream.flush()
        previous,interval = mon.complete(previous,absolute)
        result['maximum_combined_rss_bytes']=max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        result['maximum_interval_s']=max(result['maximum_interval_s'],interval)
        result['maximum_swap_growth_bytes']=max(result['maximum_swap_growth_bytes'],growth)
        result['completed_sample_intervals_s'].append(interval);samples+=1

    def tick():
        deadline()
        if time.monotonic()-previous >= .2:
            observe()

    def interrupted(*unused):
        raise RuntimeError('Owned aligned-PUSH runtime experiment interrupted')

    handlers={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        stream=(out/'samples.jsonl').open('x');observe()
        versions={k:importlib.metadata.version(k) for k in ('numpy','flopscope','whestbench','threadpoolctl')}
        require(versions==dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0'),
            'Unchanged official package versions, metadata only')
        result['versions']=versions
        tick()
        for p,digest in {**own,**inputs}.items():
            require(h.sha(p,tick)==digest,'Exact reviewed source or control input: '+str(p))
        r,e,f=[h.read(CONTROL/name) for name in ('result.json','completion.json','finalization.json')]
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256']==f['result_sha256']==CONTROL_PINS[CONTROL/'result.json']
            and f['completion_sha256']==CONTROL_PINS[CONTROL/'completion.json'], 'Actual CD production three seals')
        require(r['schema']=='current_comp_partial_d6_production_watch_v1' and r['candidate_sha256']==CD_SHA
            and r['full_model_calls']==5 and r['initialization_flops']==INIT
            and all(r.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
                'whole_owned_tree_retired','child_execution_completed','all5_complete'))
            and r['maximum_combined_rss_bytes']<RSS and r['maximum_interval_s']<=1
            and r['maximum_swap_growth_bytes']==0,'Successful original production control')
        control_rows=[r['observed_records'][i] for i in (0,4)]
        require([x['flops'] for x in control_rows]==[464400020054,462455049756]
            and all(x['rank']==35 for x in control_rows),'Exact cold/warm complete control bills')
        control_output=NEW/'current_comp_partial_d6_production_v1_0_r35.f32'
        expected=r['artifacts'][str(control_output)]
        require(expected==control_rows[0]['output_sha256']==control_rows[1]['output_sha256']
            and h.sha(control_output,tick)==expected,'Actual unchanged control output bytes')
        inputs[control_output]=expected
        manifest=h.read(ROW.parent.parent.parent/'manifest.json')['roles']['observer_fit']['row_manifests'][35]
        require(manifest['role_rank']==35 and
            (ROW.parent.parent.parent/Path(manifest['path']).parent/'weights.npy').resolve()==(ROW/'weights.npy').resolve(),
            'Actual original client fixture resolution')
        proofs={}
        for name in ('client','runner'):
            module=load(HERE/(name+'.py'),'aligned_push_'+name+'_source')
            source,proofs[name]=module.source_and_proof()
            compile(source,str(HERE/(name+'.py'))+':inert_source_review','exec');tick()
        builder=load(HERE/'build.py','aligned_push_builder')
        built,seals=builder.build(tick)
        candidate_sha=built['candidate_sha256']
        require(built['proof']['existing_complete_initialization_flops']==INIT,'Same paid startup')
        result.update(candidate_sha256=candidate_sha,build_seals=seals,adapter_proofs=proofs)
        for name in ('estimator.py','result.json','completion.json','finalization.json'):
            p=HERE/'build_v1'/name;result['artifacts'][str(p)]=h.sha(p,tick)
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'No numerical import before launch')
        argv=[str(OFFPY),'-B','-u',str(HERE/'runner.py'),'--run','--candidate',CANDIDATE,'--tag',TAG]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            FLOPSCOPE_MAX_ARRAY_BYTES='4294967296',PYTORCH_ENABLE_MPS_FALLBACK='0')
        env.update({k:'1' for k in BLAS})
        log=(out/'runner.log').open('x');reader=(out/'runner.log').open()
        child=subprocess.Popen(argv,cwd=NEW,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result.update(child_pid=child.pid,command=argv)
        h.write(out/'started.json',result)
        while child.poll() is None:
            observe();time.sleep(.15)
        observe()
        require(child.wait(timeout=1)==0 and result['full_model_calls']==2,'Exactly two successful contexts')
        mon.cleanup([child],time.monotonic()+.5)
        measured,usage_record=h.read(series),h.read(usage)
        rows=result['observed_records']
        require(measured['status']=='completed' and measured['source_sha256']==candidate_sha
            and measured['candidate']==usage_record['candidate']==CANDIDATE
            and measured['records']==rows and measured['repeated_output_byte_identical'], 'Own complete original-client outputs and bills')
        require(rows[0]['output_sha256']==rows[1]['output_sha256']
            and rows[0]['flops']-rows[1]['flops']==INIT,'Own exact repeat and once-only complete startup')
        reference=struct.unpack('<16384f',control_output.read_bytes())
        require(all(math.isfinite(x) for x in reference),'Finite saved control')
        fidelity=[]
        for row,path,base in zip(rows,outputs,control_rows):
            values=struct.unpack('<16384f',path.read_bytes())
            require(all(math.isfinite(x) for x in values),'All finite actual served values')
            drift=[abs(a-b) for a,b in zip(values,reference)]
            layer_max=[max(drift[j*1024:(j+1)*1024]) for j in range(16)]
            fidelity.append(dict(index=row['index'],all16_max_abs=max(drift),per_layer_max_abs=layer_max,
                own_complete_flops=row['flops'],control_complete_flops=base['flops'],
                complete_cost_ratio=row['flops']/base['flops']))
            require(max(drift)<=1e-5 and row['flops']<base['flops'], 'Both finite-output fidelity and strict complete cost gain')
            tick()
        del reference,values,drift
        inner=usage_record['samples']
        require(inner and 0<usage_record['peak_combined_rss_bytes']==max(x[1] for x in inner)<RSS,
            'Original leader-only check also below8GB')
        for p in (series,usage,server_log,*outputs):
            result['artifacts'][str(p)]=h.sha(p,tick)
        result['input_bindings'].update({str(p):v for p,v in inputs.items()})
        for p,digest in {**result['source_bindings'],**result['input_bindings'],**result['artifacts']}.items():
            require(h.sha(p,tick)==digest,'Fresh final direct source/input/output integrity')
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            whole_owned_tree_retired=True,child_execution_completed=True,initialization_flops=INIT,
            repeat_byte_exact=True,fidelity=fidelity,runtime_falsifier_passed=True,
            actual_completed_samples=True,foreign_numerical_workers_checked=True,
            advancement_scope='Eligible for separately reviewed CHECK4; no raw or score evidence')
        observe()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            if child is not None: mon.cleanup([child],time.monotonic()+.5)
            mon.retire_pending(time.monotonic()+.5)
            result.update(cleanup_passed=True,whole_owned_tree_retired=True,observational_groups_retired=True)
        except BaseException:
            result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for handle in (reader,log,stream):
            if handle is not None: handle.close()
        result.update(elapsed_s=time.monotonic()-began,observation_samples=samples,initial_swap_bytes=swap0,
            last_completed_observation_elapsed_s=previous-began,post_observation_memory_sample_claimed=False)
        if result['elapsed_s']>=CAP or time.monotonic()-last_progress>=120.:
            result.update(passed=False,completed=False,error='Whole/progress deadline including cleanup')
        for name in ('started.json','runner.log','samples.jsonl'):
            p=out/name
            if p.exists(): result['artifacts'][str(p)]=h.sha(p)
        if time.monotonic()-previous>1.:
            result.update(passed=False,completed=False,error='Terminal cleanup/hash interval>1s')
        h.write(out/'result.json',result);rh=h.sha(out/'result.json')
        ok=bool(result['passed'] and time.monotonic()-began<CAP and time.monotonic()-last_progress<120.
            and time.monotonic()-previous<=1.)
        h.write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh));ch=h.sha(out/'completion.json')
        elapsed=time.monotonic()-began;terminal_interval=time.monotonic()-previous
        ok=bool(ok and elapsed<CAP and time.monotonic()-last_progress<120. and terminal_interval<=1.)
        h.write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,
            elapsed_s=elapsed,terminal_publication_interval_s=terminal_interval))
        h.sha(out/'finalization.json');ok=bool(ok and time.monotonic()-began<CAP
            and time.monotonic()-last_progress<120. and time.monotonic()-previous<=1.)
        for sig,handler in handlers.items(): signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','full_model_calls',
        'maximum_combined_rss_bytes','maximum_interval_s','maximum_swap_growth_bytes','fidelity','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())
