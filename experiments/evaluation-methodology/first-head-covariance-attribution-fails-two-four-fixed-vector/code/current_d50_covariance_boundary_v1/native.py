"""Inert, paid two-layer native covariance-circuit attribution; no scoring.

The original last predictor is truncated after the L1 head. Only its L1
covariance return can be replaced by the separately acquired common-input
Torch circuit. Observers retain references; all materialization is paid.
"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD, BR = NEW/'birth_direction_fit_v1', NEW/'current_d50_native_p_bridge_v1'
CO, NG, IG = (NEW/x for x in ('current_bd_p_coadaptation_v1',
    'current_d50_native_g0_forward_v1', 'current_d50_initial_gram_witness_v1'))
OFFPY = P2/'data_gate/.venv'
PARENT = BD/'build_terminal_v1/estimator.py'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
PROTOCOL_SHA = '26b60da71938e72025c2d2a2ca9634e65aa8bd4f0e3e8827333d67b356848179'
NG_SHA = '00f7a37825577ef083b8a5eac1f230183bcbae7f9b1f2a68739e4f37d52b7049'
CPU_PROBE_SHA = '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607'
BR_JUDGE_SHA = 'a8ff76e5d020a59da6626d52b79993c9e28da118abdc458ff9bbcc1c9372f4f5'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
VERIFIER_SHA = '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608'
CASES = ('null', 'plus025', 'minus025', 'plus050', 'minus050')
SCHEMA = 'current_d50_covariance_boundary_v1'
BUDGET, RSS = 2**41, 4_000_000_000
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def patched_source():
    source = PARENT.read_text()
    nodes = {n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    lines = source.splitlines(keepends=True)
    edits = {
        '_predict_binary': (
            ('for l in range(depth):','for l in range(2):'),
            ('Cpre, Bpre = _cov_compact(w32, C, previous, previous_alive)',
             'Cpre, Bpre = _cb_cov_switch(w32, C, previous, previous_alive)'),
            ('if l == depth - 1:\n            break','if l == 1:\n            break'),
        ),
        '_predict_learned_compact': (
            ('        return fnp.asarray(delta,dtype=fnp.float64)',
             '        _cb_head_capture(l,m,delta)\n        return fnp.asarray(delta,dtype=fnp.float64)'),
        ),
    }
    changed, inverse, records = source, source, {}
    for name, changes in edits.items():
        n = nodes[name]
        old = ''.join(lines[n.lineno-1:n.end_lineno])
        new = old
        for a,b in changes:
            require(new.count(a)==1,'Unique active source edit: '+a)
            new = new.replace(a,b)
        restored = new
        for a,b in reversed(changes):
            require(restored.count(b)==1,'Unique active inverse: '+b)
            restored = restored.replace(b,a)
        require(restored==old and changed.count(old)==1,'Exact full active-function inverse')
        changed = changed.replace(old,new)
        records[name] = dict(line=n.lineno,end_line=n.end_lineno,
            original_sha256=hashlib.sha256(old.encode()).hexdigest(),
            diagnostic_sha256=hashlib.sha256(new.encode()).hexdigest(),source_inverse_byte_exact=True)
    inverse = changed
    for name, changes in reversed(tuple(edits.items())):
        n = nodes[name]; old = ''.join(lines[n.lineno-1:n.end_lineno]); new = old
        for a,b in changes: new=new.replace(a,b)
        require(inverse.count(new)==1,'Exact complete-source inverse span')
        inverse=inverse.replace(new,old)
    require(inverse==source,'Exact complete d50 source inverse')
    return changed,records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--stage',choices=('n','controls'))
    for flag in ('source-sha256','protocol-sha256','cpu-source-sha256','cpu-result-sha256','native-result-sha256'):
        parser.add_argument('--'+flag)
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(args.stage and args.source_sha256 and args.protocol_sha256==PROTOCOL_SHA,'Reviewed fixed stage/protocol')
    require(Path(sys.prefix).resolve()==OFFPY.resolve(),'OFFPY interpreter only')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','nice10/no bytecode')
    require(all(os.environ.get(k)=='1' for k in BLAS),'Explicit BLAS1')
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0','No fallback')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),'Inert admission')
    out = HERE/('native_n_v1' if args.stage=='n' else 'native_controls_v1')
    require(not out.exists(),'Exclusive native stage')
    pins = {}

    def pin(path,expected=None):
        path = Path(path).resolve(); digest=sha(path)
        require(expected is None or digest==expected,'Input/source changed: '+str(path))
        require(str(path) not in pins or pins[str(path)]==digest,'Conflicting binding')
        pins[str(path)] = digest
        return digest

    for path,digest in ((Path(__file__),args.source_sha256),(HERE/'PROTOCOL.md',PROTOCOL_SHA),
            (NG/'model.py',NG_SHA),(PARENT,PARENT_SHA),(BD/'native_v2.py',NATIVE_SHA),
            (BD/'qualify_native_v2.py',VERIFIER_SHA)):
        pin(path,digest)
    model = module(NG/'model.py','cb_frozen_inert_model_owner')
    for path,digest in model.authenticate().items(): pin(path,digest)
    native = module(BD/'native_v2.py','cb_frozen_meter_lifecycle')
    verifier = module(BD/'qualify_native_v2.py','cb_frozen_meter_verifier')
    verifier.verify_meter(pin)
    for path,digest in native.PINS.items(): pin(path,digest)
    probe_path,br_path=CO/'stage_cpu_forward_v1/result.json',BR/'judgment_v1/result.json'
    pin(probe_path,CPU_PROBE_SHA);pin(br_path,BR_JUDGE_SHA)
    probe,old=read(probe_path),read(br_path)
    require(probe['passed'] and probe['completed'] and probe['no_fit'],'Complete original sealed P probe')
    require(old['passed'] is False and old['completed'] is False and old['null_bytes_and_bills_exact']
            and not old['truth_accessed'] and set(old['native_records'])==set(CASES),'Old bridge stays failed; native inventory only')
    for doc in (probe,old):
        for group in ('source_bindings','input_bindings','artifacts'):
            for path,value in doc.get(group,{}).items():
                if isinstance(value,dict): pin(value['path'],value['sha256'])
                else: pin(path,value)
    baseline = {}
    for case in CASES:
        item = old['native_records'][case]
        rp,wp = BR/f'run_{case}_v1/result.json',BR/f'watch_{case}_v1/result.json'
        pin(rp,item['result_sha256']); pin(wp,item['watch_sha256'])
        r,w = read(rp),read(wp)
        require(item['complete_loop_verified'] and r['passed'] and r['hashes_unchanged']
                and w['passed'] and w['completed'] and not w['terminated']
                and w['child_result_sha256']==item['result_sha256'],'Exact completed native loop')
        for path,digest in r['artifacts'].items(): pin(path,digest)
        baseline[case] = {int(t['rank']):t for t in r['trials'][:2]}
        pin(probe['payloads'][case]['path'],probe['payloads'][case]['sha256'])
    ig=read(IG/'acquisition_v1/result.json')
    row_info={}
    for rank in (35,100):
        row=next(t for t in ig['trials'] if t['rank']==rank)
        # Canonical row identities come from the original manifest, not names.
        manifest=read(native.STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
        entry=next(x for x in manifest if x['role_rank']==rank)
        rp=native.STORE/entry['path'];pin(rp,entry['sha256']); meta=read(rp)
        wp=rp.parent/'weights.npy';pin(wp,meta['files']['weights.npy']['sha256'])
        require(str(wp) in ig['input_bindings'] and pins[str(wp)]==ig['input_bindings'][str(wp)]
                and meta['role']=='observer_fit' and meta['role_rank']==rank,'Same canonical weights as paid G0')
        row_info[rank]=dict(weights=str(wp),G0=row['G0'])

    def sealed(stage,digest):
        rp=HERE/stage/'result.json';cp=rp.parent/'completion.json'
        require(digest and len(digest)==64,'Reviewed prior result hash')
        pin(rp,digest);pin(cp);r,c=read(rp),read(cp)
        require(r['schema']==SCHEMA and r['passed'] and r['completed'] and r['hashes_unchanged']
                and not r['truth_accessed'] and c['passed'] and c['completed'] and c['result_sha256']==digest,'Complete prior stage')
        for group in ('source_bindings','input_bindings','artifacts'):
            for path,value in r[group].items(): pin(path,value)
        return r

    cpu=nrecord=None
    if args.stage=='controls':
        require(all(isinstance(x,str) and len(x)==64 for x in
            (args.cpu_source_sha256,args.cpu_result_sha256,args.native_result_sha256)),
            'Reviewed CPU/source and both previous result hashes required')
        pin(HERE/'cpu.py',args.cpu_source_sha256)
        cpu=sealed('cpu_v1',args.cpu_result_sha256)
        nrecord=sealed('native_n_v1',args.native_result_sha256)
        require(cpu['stage']=='cpu' and nrecord['stage']=='n','Prior stage ownership')
    for key,value in {'numpy':'2.4.6','flopscope':'0.12.0','whestbench':'0.16.0'}.items():
        require(importlib.metadata.version(key)==value,'Pinned version: '+key)
    source,source_records=patched_source()
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),'All admission before numeric imports')
    out.mkdir()
    result=dict(schema=SCHEMA,stage=args.stage,passed=False,completed=False,hashes_unchanged=False,
        truth_accessed=False,training_targets_accessed=False,parameter_updates=0,gradient_calls=0,
        full_card_calls=0,prefix_calls=0,no_fit=True,score_measured=False,MPS_used=False,
        source_bindings=dict(pins),input_bindings=dict(pins),artifacts={},ranks={},trials=[],
        versions=dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0'),
        blas_environment={k:os.environ[k] for k in BLAS},
        source_inverse_byte_exact=True,source_records=source_records,
        diagnostic_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        repeated_payload_binding_is_diagnostic=True,complete_inference_bill_claimed=False,
        started_utc=datetime.now(timezone.utc).isoformat())
    native.write(out/'started.json',result)
    start=time.monotonic()
    try:
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        flops.configure(symmetry_warnings=False)
        env={'__name__':'cb_native_owned_prefix','__file__':str(PARENT)}
        exec(compile(source,str(PARENT)+':cb-prefix','exec'),env)
        require(env['_PACKED_ELEMENTS']==292225024 and env['_PACKED_TILE']==8192
                and env['_square_schedule'](1024,1024,1024)==(5,(1024,1024,1024)),'Actual current D5 arena/schedule')
        original_cov,original_select,original_gaussian=env['_cov_compact'],env['select'],env['gaussian_raw']
        original_first=env['_nonlinear_first_gaussian']
        capture,mode,supplied={},'N',None

        def cov_switch(w,C,indices,alive):
            capture.update(C0=C,W1=w,indices=indices,alive=alive)
            pair=original_cov(w,C,indices,alive) if mode=='N' else supplied
            capture.update(Cpre=pair[0],Bpre=pair[1])
            return pair

        def head_capture(l,host,delta):
            if l==1: capture.update(host1=host,delta1=delta)

        def select_observer(score,floor):
            pair=original_select(score,floor)
            capture.setdefault('selectors',[]).append(pair)
            return pair

        def gaussian_observer(mu,v):
            capture.setdefault('prevariances',[]).append(v)
            return original_gaussian(mu,v)

        def first_observer(*args,**kwargs):
            capture['G0']=args[1]
            return original_first(*args,**kwargs)

        env.update(_cb_cov_switch=cov_switch,_cb_head_capture=head_capture,
                   select=select_observer,gaussian_raw=gaussian_observer,
                   _nonlinear_first_gaussian=first_observer)
        with np.load(NEW/'payload_response_rows_v1.npz',allow_pickle=False) as data:
            original_payload={k:data[k].copy() for k in data.files}
        literals={};payload_bytes={}
        for case in CASES:
            with np.load(probe['payloads'][case]['path'],allow_pickle=False) as data:
                require(set(data.files)==set(original_payload),'Complete sealed payload keys')
                values={k:data[k].copy() for k in data.files}
            require(all(v.dtype==np.float32 and np.isfinite(v).all() for v in values.values()),'Exact finite FP32 payload')
            literals[case]={k:v.tolist() for k,v in values.items()}
            payload_bytes[case]={k:v.tobytes() for k,v in values.items()}
        sequence=[(r,'N',c,False) for r in (35,100) for c in CASES]+[(35,'N','null',True)] if args.stage=='n' else [
            (r,m,c,False) for r in (35,100) for m in ('L','R') for c in CASES]+[(35,'L','null',True),(35,'R','null',True)]
        result['sequence']=[dict(rank=r,circuit=m,case=c,repeat=q) for r,m,c,q in sequence]
        saved_first={}

        def save(path,value):
            with path.open('xb') as stream: np.save(stream,value,allow_pickle=False)
            info=dict(path=str(path),sha256=sha(path),dtype=str(value.dtype),shape=list(value.shape),
                      raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())
            result['artifacts'][str(path)]=info['sha256']
            return info

        def loaded(info):
            require(sha(info['path'])==info['sha256'],'Array seal')
            return np.load(info['path'],allow_pickle=False)

        with threadpool_limits(limits=1):
            result['threadpools']=threadpool_info()
            require(all(p['num_threads']==1 for p in result['threadpools'] if p.get('user_api')=='blas'),'Observed BLAS1')
            result['empty_threadpool_inventory_is_unknown']=not result['threadpools']
            for call,(rank,mode,case,repeat) in enumerate(sequence):
                capture={};supplied=None
                weights=np.load(row_info[rank]['weights'],mmap_mode='r',allow_pickle=False)
                require(weights.shape==(16,1024,1024) and weights.dtype==np.float32,'Canonical weights')
                arrays=None if mode=='N' else cpu['ranks'][str(rank)]['circuits'][mode]
                inputs=None if arrays is None else (loaded(arrays['Cpre']),loaded(arrays['Bpre']))
                if inputs is not None:
                    require(all(x.dtype==np.float64 and x.shape==(1024,1024) and np.isfinite(x).all() for x in inputs)
                            and inputs[0].tobytes()==inputs[1].tobytes(),'Owned same-input full-width Cpre/Bpre')
                tag=f'call_{call:02d}_rank{rank}_{mode}_{case}'
                result['prefix_calls']+=1
                begin=time.monotonic()
                with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as ctx:
                    owned=[fnp.array(weights[l],dtype=fnp.float32) for l in (0,1)]
                    if inputs is not None:
                        supplied=tuple(fnp.array(x,dtype=fnp.float64) for x in inputs)
                    # Literal bindings contain only the already sealed payload.
                    # Each rebind's FNP initialization is included in this bill.
                    env['_GNN_LITERAL']=literals[case];env['_GNN_PARAMS']=None
                    pred=env['_predict_learned_compact'](SimpleNamespace(width=1024,depth=16,weights=owned))
                    names=('C0','W1','indices','alive','Cpre','Bpre','host1','delta1','G0')
                    packed={name:(capture[name].tobytes(),str(capture[name].dtype),tuple(capture[name].shape)) for name in names}
                    pred_data=pred.tobytes()
                    variances=[v.tobytes() for v in capture['prevariances']]
                    selectors=[(i.tobytes(),str(i.dtype),tuple(i.shape),m.tobytes(),str(m.dtype)) for i,m in capture['selectors']]
                    payload_exact=all(env['_GNN_PARAMS'][k].tobytes()==payload_bytes[case][k] for k in literals[case])
                trial=dict(call=call,rank=rank,circuit=mode,case=case,repeat=repeat,
                    **native.metrics(ctx,begin),owned_weight_layers=[0,1],
                    includes_all_diagnostic_input_copies=True,payload_sha256=probe['payloads'][case]['sha256'])
                decoded={name:np.frombuffer(data,dtype=dtype).reshape(shape) for name,(data,dtype,shape) in packed.items()}
                prediction=np.frombuffer(pred_data,dtype=np.float32).reshape(2,1024)
                entry={name:save(out/f'{tag}_{name}.npy',v) for name,v in decoded.items()}
                entry['prediction']=save(out/f'{tag}_prediction.npy',prediction)
                diagnostics=dict(prevariances=[dict(minimum=float(np.frombuffer(v,dtype=np.float64).min()),
                    finite=bool(np.isfinite(np.frombuffer(v,dtype=np.float64)).all()),
                    below_floor_count=int((np.frombuffer(v,dtype=np.float64)<1e-10).sum())) for v in variances],
                    selectors=[dict(ordered_ids=np.frombuffer(i,dtype=dt).reshape(shape).tolist(),
                        membership=np.flatnonzero(np.frombuffer(m,dtype=md)).tolist(),
                        count=shape[0],mask_sha256=hashlib.sha256(m).hexdigest()) for i,dt,shape,m,md in selectors],
                    payload_byte_exact=payload_exact,native_old_two_rows_byte_exact=None,
                    first_push_all_active=True,full_width_square_schedule=[5,[1024,1024,1024]])
                entry['diagnostics']=diagnostics
                bucket=result['ranks'].setdefault(str(rank),dict(arrays={},outputs={},circuits={}))
                if args.stage=='n':
                    if repeat: result['repeat']=entry
                    else:
                        bucket['outputs'][case]=entry
                        if case=='null': bucket['arrays']={k:entry[k] for k in names[:6]}
                else:
                    if repeat: result.setdefault('repeats',{})[mode]=entry
                    else: bucket['circuits'].setdefault(mode,dict(outputs={}))['outputs'][case]=entry
                result['trials'].append(trial)
                bill=out/f'{tag}_bill.json';native.write(bill,trial);result['artifacts'][str(bill)]=sha(bill)
                attempt=out/f'{tag}_attempt.json';native.write(attempt,entry);result['artifacts'][str(attempt)]=sha(attempt)
                ref=weakref.ref(ctx);life=native.reset_saved_history(flops,ctx,bill);del ctx;gc.collect()
                life.update(context_released=ref() is None,complete=True,call=call,rank=rank)
                lp=out/f'{tag}_lifecycle.json';native.write(lp,life);result['artifacts'][str(lp)]=sha(lp)
                native.guard(trial)
                require(trial['peak_rss_bytes']<RSS and time.monotonic()-start<600,'Strict diagnostic resource guard')
                require(life['context_released'] and life['closed_context_bill_unchanged_by_reset'],'Paid closed-context lifecycle')
                require(pred.dtype==np.float32 and pred.shape==(2,1024) and np.isfinite(prediction).all()
                        and (prediction>=0).all() and all(np.isfinite(x).all() for x in decoded.values()),'Finite native output contract')
                require(len(variances)==2 and all(d['finite'] and d['minimum']>=1e-10 and d['below_floor_count']==0 for d in diagnostics['prevariances']), 'Two positive unfloored domains')
                require(len(selectors)==5 and all(sorted(d['ordered_ids'])==d['membership'] for d in diagnostics['selectors'])
                        and all(d['count']==1024 and d['ordered_ids']==list(range(1024)) for d in diagnostics['selectors'][:3]),'Exact first-layer and partial selector roster')
                require(decoded['indices'].tolist()==list(range(1024)) and decoded['alive'].all()
                        and decoded['C0'].dtype==np.float64 and decoded['Cpre'].dtype==np.float64
                        and decoded['Bpre'].tobytes()==decoded['Cpre'].tobytes()
                        and decoded['W1'].tobytes()==weights[1].tobytes() and payload_exact,'Canonical common covariance input and payload')
                require(decoded['G0'].tobytes()==loaded(row_info[rank]['G0']).tobytes(),
                        'Recomputed initial Gram exactly matches sealed weights-only witness')
                if mode=='N':
                    old_values=np.load(baseline[case][rank]['output'],allow_pickle=False)
                    exact=prediction.tobytes()==old_values[:2].tobytes()
                    diagnostics['native_old_two_rows_byte_exact']=exact
                    require(exact,'Literal native prefix must match sealed bank L0/L1 bytes')
                else:
                    for key in names[:4]:
                        require(decoded[key].tobytes()==loaded(nrecord['ranks'][str(rank)]['arrays'][key]).tobytes(),'Controls identical common input: '+key)
                if case=='null' and not repeat:
                    saved_first[(rank,mode)]={k:v[0] for k,v in packed.items()}
                    saved_first[(rank,mode)]['prediction']=pred_data
                if repeat:
                    repeated={k:v[0] for k,v in packed.items()};repeated['prediction']=pred_data
                    diagnostics['repeat_byte_exact']=repeated==saved_first[(rank,mode)]
                    require(diagnostics['repeat_byte_exact'],'Exact repeated two-layer output')
                if args.stage=='n' and case!='null':
                    for key in names[:6]:
                        require(decoded[key].tobytes()==loaded(bucket['arrays'][key]).tobytes(),'P-independent native boundary: '+key)
                print(json.dumps(dict(stage=args.stage,call=call,rank=rank,circuit=mode,case=case,repeat=repeat,
                    flops=trial['flops'],wall_s=trial['wall_s'],passed=True)),flush=True)
                del weights,owned,pred,packed,decoded,prediction,inputs,arrays,capture,supplied
                capture={};supplied=None;gc.collect()
        require(result['prefix_calls']==(11 if args.stage=='n' else 22),'Complete fixed native sequence')
        require(all(sha(p)==h for p,h in pins.items()),'Unchanged inputs and sources')
        require(all(sha(p)==h for p,h in result['artifacts'].items()),'Unchanged artifacts')
        require('torch' not in sys.modules,'No Torch/MPS native import')
        result.update(passed=True,completed=True,hashes_unchanged=True,
            native_baseline_reproduces_all_old_two_rows=(args.stage=='n'),repeat_byte_exact=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,error=traceback.format_exc())
    result.update(elapsed_s=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat())
    native.write(out/'result.json',result)
    native.write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('stage','passed','completed','prefix_calls','elapsed_s','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
