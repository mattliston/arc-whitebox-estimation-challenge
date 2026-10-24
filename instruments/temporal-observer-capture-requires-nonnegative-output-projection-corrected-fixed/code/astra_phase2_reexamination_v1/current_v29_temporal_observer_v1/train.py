"""DRAFT: one fixed cached projected-observer proposal; inert without --run."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import types
import weakref

HERE=Path(__file__).resolve().parent
NEW,P2=HERE.parent,HERE.parent.parent
AP='/Users/matt/anaconda3/bin/python'
SITE=Path('/Users/matt/anaconda3/lib/python3.11/site-packages')
TRUTH=P2/'public_g22_operator_features_v1/truthpost'
P1=NEW/'current_d50_native_objective_p1_v1'
INDEX=NEW/'joint_prefix_final_fit_v1/manifest.json'
SCHEMA='current_v29_temporal_observer_training_v1'
MODEL_SHA='13a5e562561c324709a31ccfab4e53b1f7ac10079020459df098bd453802a138'
TRAIN=(35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
CHECK=(37,39,42,48)
CAP,RSS,DRIVER=600.,12_000_000_000,16_000_000_000
BLAS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
      'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
CAPTURE_FLAGS=('capture-worker','capture-watch-source','capture-protocol','capture-watch-spec',
    'capture','capture-watch','projection-qualification','projection-qualification-watch')
EXTRA_FLAGS=('neural-source','neural-watch-source','neural-protocol','neural-watch-spec',
    'neural','neural-watch','check-capture','check-capture-watch','joint-inventory')+CAPTURE_FLAGS
FLAGS=('source','watcher','protocol','watch-spec','model')+EXTRA_FLAGS
EXTERNAL={SITE/name for name in ('torch/optim/adam.py','torch/optim/adamw.py',
    'torch/__init__.py','numpy/__init__.py','torch/nn/utils/clip_grad.py')}
CLIP_SHA='2e363762653c9ea5204f96467594200705683ac619ce24c47bb101b523ea0eb1'
INPUT_PINS={INDEX:'d61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971',
    P1/'training_v1/result.json':'aaaef3d5f5a3c60583af779895b8db4f769276c64cb0cc032e93755169dbab91',
    P1/'training_v1/completion.json':'6964011b2262a70e56fafd74b040432065513706c3fde5586341a09b85fd0253'}


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def valid(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def sha(path,tick=None):
    p=Path(path); require(not p.is_symlink(),'No symlink'); p=p.resolve()
    require((p.is_relative_to(P2) or p in EXTERNAL) and 'truthpost' not in p.parts
        and not (p.suffix in ('.npy','.npz') and 'truth' in p.name.lower()), 'Nontarget hash boundary')
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            digest.update(block)
            if tick is not None:
                tick()
    return digest.hexdigest()


def hash_target(path,digest,targets,tick=None):
    """Separate exact20 TRAIN byte allowlist; never a CHECK target decoder."""
    p=Path(path); require(not p.is_symlink(),'No target symlink'); p=p.resolve()
    allowed={Path(v['path']):v['sha256'] for v in targets.values()}
    require(len(allowed)==20 and p in allowed and allowed[p]==digest
        and p in {TRUTH/f'truthpost_rank{r:04d}.npz' for r in TRAIN},'Only original20 TRAIN labels')
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick is not None:
                tick()
    require(h.hexdigest()==digest,'Exact original TRAIN target bytes')
    return digest


def read(path):
    p=Path(path); require(not p.is_symlink() and p.stat().st_size<16*1024**2,'Bounded JSON')
    return json.loads(p.read_text())


def write(path,value):
    def safe(x):
        if isinstance(x,float) and not math.isfinite(x):
            return dict(nonfinite_float=repr(x))
        if isinstance(x,dict):
            return {k:safe(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):
            return [safe(v) for v in x]
        return x
    with Path(path).open('x') as stream:
        json.dump(safe(value),stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def add(mapping,path,digest,tick=None):
    path=str(Path(path).resolve())
    require(valid(digest) and mapping.get(path,digest)==digest and sha(path,tick)==digest,'Immutable '+path)
    mapping[path]=digest


def load(path,name,digest):
    require(sha(path)==digest,'Reviewed inert source')
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


def inventory(record,sources,inputs,tick=None):
    for group in ('source_bindings','input_bindings','weight_bindings','artifacts'):
        for path,digest in record.get(group,{}).items():
            add(sources if group=='source_bindings' else inputs,path,digest,tick)


def qualification_args(args):
    values={k.replace('-','_')+'_sha256':getattr(args,k.replace('-','_')+'_sha256') for k in CAPTURE_FLAGS}
    values.update(source_sha256=args.neural_source_sha256,watcher_sha256=args.neural_watch_source_sha256,
        protocol_sha256=args.neural_protocol_sha256,watch_spec_sha256=args.neural_watch_spec_sha256,model_sha256=args.model_sha256)
    return types.SimpleNamespace(**values)


def owners(args):
    require(all(valid(getattr(args,k.replace('-','_')+'_sha256',None)) for k in FLAGS)
        and args.model_sha256==MODEL_SHA,'All reviewed source and actual prerequisite hashes')
    return {HERE/'train.py':args.source_sha256,HERE/'watch_train.py':args.watcher_sha256,
        HERE/'TRAIN_PROTOCOL.md':args.protocol_sha256,HERE/'TRAIN_WATCH_SPEC.md':args.watch_spec_sha256,
        HERE/'projected_model.py':args.model_sha256,HERE/'neural_qualification_v2.py':args.neural_source_sha256,
        HERE/'watch_neural_qualification_v2.py':args.neural_watch_source_sha256,
        HERE/'NEURAL_QUALIFICATION_V2_PROTOCOL.md':args.neural_protocol_sha256,
        HERE/'NEURAL_QUALIFICATION_V2_WATCH_SPEC.md':args.neural_watch_spec_sha256,
        SITE/'torch/nn/utils/clip_grad.py':CLIP_SHA}


def cached_loader(q):
    """Qualified64-array loader, one exact index-owner substitution for CHECK."""
    source,_=q.derive_source(); nodes=ast.parse(source).body; lines=source.splitlines(keepends=True)
    selected=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='cached_data']
    require(len(selected)==1,'Unique qualified cached loader')
    n=selected[0]; original=''.join(lines[n.lineno-1:n.end_lineno])
    old='call = TRAIN.index(rank); record = capture[\'capture_records\'][call]'
    new='call = capture[\'ranks\'].index(rank); record = capture[\'capture_records\'][call]'
    require(original.count(old)==1 and new not in original,'Only explicit capture roster changes')
    changed=original.replace(old,new); require(changed.replace(new,old)==original,'Whole loader inverse')
    scope=dict(vars(q.load_worker()))
    exec(compile(changed,str(HERE/'neural_qualification_v2.py')+':explicit-capture-index','exec'),scope)
    return scope['cached_data'],dict(owner_sha256=sha(HERE/'neural_qualification_v2.py'),whole_source_inverse=True,
        original_packet_sha256=hashlib.sha256(original.encode()).hexdigest(),derived_packet_sha256=hashlib.sha256(changed.encode()).hexdigest())


def admit(args,sources,inputs,weights,tick=None):
    for path,digest in owners(args).items():
        add(sources,path,digest,tick)
    q=load(HERE/'neural_qualification_v2.py','training_qualified_neural_metadata',args.neural_source_sha256)
    qa=qualification_args(args); admitted=q.admit(qa,sources,inputs,weights,tick)
    capture=load(HERE/'projection_capture_worker.py','training_capture_metadata',args.capture_worker_sha256)
    h=capture._helpers()
    qr,qs=h['chain'](HERE/'neural_qualification_v2',args.neural_sha256,inputs,180.,tick)
    qw,qws=h['chain'](HERE/'neural_qualification_watch_v2',args.neural_watch_sha256,inputs,180.,tick)
    q.verify_result(qr,qa,admitted,tick)
    require(qw['schema']=='current_v29_temporal_observer_neural_qualification_watch_v2'
        and all(qw.get(k) is True for k in ('passed','completed','cleanup_passed','whole_owned_tree_retired',
            'observational_groups_retired','sources_unchanged','hashes_unchanged','neural_execution_qualified','actual_completed_samples'))
        and qw['retired_calls']==qw['completed_graph_count']==11
        and qw['maximum_combined_rss_bytes']<RSS and qw['maximum_interval_s']<=1
        and qw['maximum_swap_growth_bytes']==0 and qw['child_result_sha256']==qs['result_sha256']
        and qw['child_completion_sha256']==qs['completion_sha256'] and qw['child_finalization_sha256']==qs['finalization_sha256']
        and qw['prerequisite_seals']==admitted['prerequisite_seals'] and qw['model_source_proof']==admitted['model_proof']
        and qw['no_postpublication_observations'] and qw['publication_tail_elapsed_before_result_s']<1,
        'Whole new neural qualification and independent resource/lineage authority')
    inventory(qr,sources,inputs,tick); inventory(qw,sources,inputs,tick)
    ca=types.SimpleNamespace(stage='check',source_sha256=args.capture_worker_sha256,
        watch_sha256=args.capture_watch_source_sha256,protocol_sha256=args.capture_protocol_sha256,
        watch_spec_sha256=args.capture_watch_spec_sha256,adapter_sha256=capture.ADAPTER_SHA,
        adapter_protocol_sha256=capture.ADAPTER_PROTOCOL_SHA,
        qualification_sha256=args.projection_qualification_sha256,qualification_watch_sha256=args.projection_qualification_watch_sha256,
        train_sha256=args.capture_sha256,train_watch_sha256=args.capture_watch_sha256)
    prior=capture.prerequisites(ca,sources,inputs,tick)
    check,cs=h['chain'](HERE/'check_projection_capture_v1',args.check_capture_sha256,inputs,660.,tick)
    cw,cws=h['chain'](HERE/'check_projection_capture_watch_v1',args.check_capture_watch_sha256,inputs,660.,tick)
    capture.verify_expanded(check,'check',ca,prior['qualification'],tick)
    h['watcher_contract'](cw,check,cs,CHECK+(37,),'check')
    inventory(check,sources,inputs,tick); inventory(cw,sources,inputs,tick)
    require(check['prerequisite_seals']==prior['seals'] and cw['all25_capture_inventory_sealed']
        and cw['projected_capture_verified'],'Actual fresh CHECK5 after sealed TRAIN20')
    jp=HERE/'check_projection_capture_watch_v1/joint_capture_inventory.json'; add(inputs,jp,args.joint_inventory_sha256,tick)
    joint=read(jp); train=admitted['capture']
    require(cw['joint_capture_inventory']==str(jp) and cw['joint_capture_inventory_sha256']==args.joint_inventory_sha256
        and cw['artifacts'][str(jp)]==args.joint_inventory_sha256
        and joint['schema']=='current_v29_temporal_projection_joint_capture_inventory_v1'
        and joint['passed'] and joint['completed'] and joint['full_model_calls']==25 and joint['setup_calls']==2
        and joint['call_artifact_count']==225 and joint['raw_arrays_per_call']==64
        and joint['train_ranks']==list(TRAIN) and joint['check_ranks']==list(CHECK+(37,))
        and joint['raw_host_semantics']=='raw_physical_unclipped' and joint['served_semantics']=='final_output_maximum_zero'
        and joint['candidate_sha256']==train['candidate_sha256']==check['candidate_sha256']
        and joint['prerequisite_seals']==dict(prior['seals'],check=cs)
        and joint['train_worker_pid']==train['worker_pid'] and joint['check_worker_pid']==check['worker_pid']
        and joint['train_worker_pid']!=joint['check_worker_pid'] and joint['two_serial_fresh_processes']
        and joint['train_prior_process_retired'] and joint['current_check_process_retired']
        and joint['train_capture_records']==train['capture_records'] and joint['check_capture_records']==check['capture_records']
        and joint['train_complete_bills']==train['trials'] and joint['check_complete_bills']==check['trials']
        and not any(joint[k] for k in ('truth_accessed','score_measured','head_attached','parameter_updates','downstream_training_authorized')),
        'Actual complete25-call pretruth joint boundary; no imagined capture authority')
    calls={}; metadata={}
    for value,folder in ((train,HERE/'train_projection_capture_v1'),(check,HERE/'check_projection_capture_v1')):
        for path,digest in value['artifacts'].items():
            (calls if Path(path).parent==folder and Path(path).name.startswith('call_') else metadata)[path]=digest
    require(joint['call_artifacts']==calls and len(calls)==225 and joint['stage_metadata_artifacts']==metadata
        and len(metadata)==4,'Exact joint call/artifact union')
    fixtures=dict(train['fixture_bindings']); fixtures.update(check['fixture_bindings'])
    require(joint['fixture_bindings']==fixtures and len(fixtures)==24,'Exactly24 canonical independent fixtures')
    inventory(joint,sources,inputs,tick)
    for value in fixtures.values():
        add(weights,value['path'],value['sha256'],tick)
    for path,digest in INPUT_PINS.items():
        add(inputs,path,digest,tick)
    old=read(P1/'training_v1/result.json'); end=read(P1/'training_v1/completion.json')
    require(all(old[k] for k in ('passed','completed','hashes_unchanged','sources_unchanged'))
        and end['passed'] and end['completed'] and end['result_sha256']==INPUT_PINS[P1/'training_v1/result.json'],
        'Completed original1B target metadata authority only, no cost/head reuse')
    index={int(v['rank']):v for v in read(INDEX)['target_sources']}; targets={}
    for rank in TRAIN:
        entry=index[rank]; path=TRUTH/f'truthpost_rank{rank:04d}.npz'; fixture=fixtures[str(rank)]
        require(entry['weight_file']==fixture['path'] and entry['target_file']==str(path)
            and old['target_bindings'][str(path)]==entry['target_sha256']
            and old['weight_bindings'][fixture['path']]==fixture['sha256'], 'Exact original20 TRAIN targets and same weights')
        targets[rank]=dict(path=str(path),sha256=entry['target_sha256'])
    loader,proof=cached_loader(q)
    require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),'All actual eligibility before labels/numerics')
    return dict(q=q,model=admitted['model'],model_proof=admitted['model_proof'],loader=loader,loader_proof=proof,
        train=train,check=check,targets=targets,joint_sha256=args.joint_inventory_sha256,neural_result=qr,
        prerequisite_seals=dict(**admitted['prerequisite_seals'],neural=qs,neural_watch=qws,check=cs,check_watch=cws),
        fixture_bindings=fixtures,candidate_sha256=joint['candidate_sha256'])


def entries(orders):
    rows=[]
    for epoch,order in enumerate(orders):
        for i,rank in enumerate(order):
            rows.append(dict(graph_index=len(rows),rank=rank,kind='train_backward',device='mps',epoch=epoch,update=epoch*5+i//4))
    for kind,ranks in (('terminal_train_forward',TRAIN),('terminal_check_forward',CHECK)):
        for rank in ranks:
            rows.append(dict(graph_index=len(rows),rank=rank,kind=kind,device='mps',epoch=None,update=None))
    return rows


def check_roster(path,stage):
    value=read(path)
    require(stage=='train' and value['schema']=='current_v29_temporal_observer_training_roster_v1'
        and value['seed']==93091230 and value['generator']=='PCG64DXSM' and len(value['orders'])==30
        and all(len(row)==20 and all(type(r) is int for r in row) and set(row)==set(TRAIN) for row in value['orders'])
        and value['entries']==entries(value['orders']) and len(value['entries'])==624,'Exact30 sealed TRAIN permutations and20+4 terminal forwards')
    return value['entries']


def rawsha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def arraymeta(value):
    return dict(dtype=str(value.dtype),shape=list(value.shape),raw_sha256=rawsha(value))


def targets_and_scale(np,admitted,result,guard):
    targets={}; ratios=[]; rows={}
    for rank in TRAIN:
        info=admitted['targets'][rank]; hash_target(info['path'],info['sha256'],admitted['targets'],guard)
        result['target_bindings'][info['path']]=info['sha256']; result['training_targets_accessed']=True
        with np.load(info['path'],allow_pickle=False) as saved:
            target=saved['mc_post'].copy()
        require(target.dtype==np.float32 and target.shape==(16,1024) and np.isfinite(target).all(),'Original finite FP32 1B label archive')
        record=admitted['train']['capture_records'][TRAIN.index(rank)]
        with np.load(record['path'],allow_pickle=False) as saved:
            host=saved['layer_15_host'].copy(); sigma=saved['layer_15_sigma'].copy()
        for key,value in (('layer_15_host',host),('layer_15_sigma',sigma)):
            require(arraymeta(value)==record['arrays'][key] and np.isfinite(value).all(),'Actual raw physical scalar inputs')
        require((sigma>0).all(),'Positive original physical sigma')
        raw=host.astype(np.float64); truth=target[15].astype(np.float64); projected=np.maximum(host,0).astype(np.float64)
        ratios.append(((truth-raw)/sigma.astype(np.float64))**2)
        rows[str(rank)]=dict(original_dtype=str(target.dtype),original_label_raw_sha256=rawsha(target),
            raw_host_final_mse=float(np.mean((raw-truth)**2)),projected_host_final_mse=float(np.mean((projected-truth)**2)))
        targets[rank]=target[15].copy(); guard()
    scalar=np.array(max(2**-20,math.sqrt(float(np.mean(np.stack(ratios))))),dtype=np.float32)
    require(np.isfinite(scalar) and scalar>0,'One frozen TRAIN-only raw-host residual scale')
    return targets,scalar,rows


def run_one(np,torch,q,model,spec,source,parameters,target=None):
    backward=spec['kind']=='train_backward'; require(backward==(target is not None),'No label supplied to final cached forwards')
    ids={k:id(v) for k,v in parameters.items()}; persistent_ids=set(ids.values())
    before={k:rawsha(q.asarray(v)) for k,v in parameters.items()}
    data={k:torch.tensor(v,dtype=torch.float32,device='mps') for k,v in source.items()}
    inputs={k:rawsha(q.asarray(v)) for k,v in data.items()}
    refs=[weakref.ref(v) for v in data.values()]; saved=[]; driver=[]
    def sample():
        torch.mps.synchronize(); value=int(torch.mps.driver_allocated_memory())
        require(0<=value<DRIVER,'Live MPS driver16GB'); driver.append(value)
    def pack(value):
        if id(value) not in persistent_ids:
            saved.append(weakref.ref(value))
        return value
    packet=dict(spec); sample()
    with torch.set_grad_enabled(backward),torch.autograd.graph.saved_tensors_hooks(pack,lambda value:value):
        out=model.forward(data['features'],data['sigma'],data['host'],data['weights'],parameters,
            data['centers'],data['scales'],data['r'])
        sample(); served=q.asarray(out['served'])
        require(served[:15].tobytes()==np.maximum(source['host'][:15],0).tobytes(),'Exact projected unaffected prefix')
        if backward:
            label=torch.tensor(target,dtype=torch.float32,device='mps'); refs.append(weakref.ref(label))
            residual=out['served'][15]-label; mse=(residual*residual).mean(); loss=(mse/2e-8)/4
            require(bool(torch.isfinite(loss).item()),'Finite exact final FP32 proposal loss')
            loss.backward(); sample()
            grad,norms=q.gradients(np,parameters,nontrivial=False)
            packet.update(FP32_final_mse=float(mse.detach().cpu()),divided_batch_loss=float(loss.detach().cpu()),
                accumulated_gradient_norms=norms,accumulated_gradient_hashes={k:rawsha(v) for k,v in grad.items()})
            refs.extend(weakref.ref(v) for v in (residual,mse,loss))
        refs.extend(weakref.ref(v) for v in (out['served'],out['delta'],out['state']))
    require(ids=={k:id(v) for k,v in parameters.items()} and before=={k:rawsha(q.asarray(v)) for k,v in parameters.items()},
        'Actual supplied leaves and parameter bytes unchanged during one rollout')
    require(inputs=={k:rawsha(q.asarray(v)) for k,v in data.items()}
        and all(not v.requires_grad and v.grad is None for v in data.values()),'Captured inputs and frozen normalizers unchanged')
    packet.update(backward=backward,parameter_hashes=before,input_hashes=inputs,finite=True,
        inputs_normalizers_parameters_unchanged=True,output_sha256=rawsha(served),
        live_mps_driver_samples=driver,maximum_live_mps_driver_bytes=max(driver),saved_tensor_references=len(saved))
    return packet,served,refs+saved


def update_once(np,torch,q,optimizer,parameters,index,lr):
    require(len(optimizer.param_groups)==1 and [id(p) for p in optimizer.param_groups[0]['params']]
        ==[id(p) for p in parameters.values()], 'One exact whole-parameter AdamW group')
    _,before_norms=q.gradients(np,parameters)
    expected=parameters['E0'][:3].detach().clone(); expected.mul_(1-lr*2e-5)
    total=torch.nn.utils.clip_grad_norm_(list(parameters.values()),5.,error_if_nonfinite=True,foreach=False)
    optimizer.step(); torch.mps.synchronize()
    require(len(optimizer.state)==10 and q.asarray(parameters['E0'][:3]).tobytes()==q.asarray(expected).tobytes(),
        'Original dead384 data rows receive only actual FP32 decoupled decay')
    states={}
    for key,p in parameters.items():
        state=optimizer.state[p]
        require(set(state)=={'step','exp_avg','exp_avg_sq'} and int(state['step'].item())==index+1
            and bool(torch.isfinite(p).all().item()) and all(bool(torch.isfinite(v).all().item()) for v in state.values()),'Exact finite actual AdamW states')
        if key=='E0':
            require(bool((p.grad[:3]==0).all().item()) and bool((state['exp_avg'][:3]==0).all().item())
                and bool((state['exp_avg_sq'][:3]==0).all().item()),'Dead384 gradients/moments remain zero')
        states[key]=dict(step=index+1,exp_avg_sha256=rawsha(q.asarray(state['exp_avg'])),exp_avg_sq_sha256=rawsha(q.asarray(state['exp_avg_sq'])))
    return dict(update=index+1,learning_rate=lr,preclip_global_norm=float(total.detach().cpu()),
        preclip_block_norms=before_norms,states=states,dead384_exact_decay=True,
        mps_driver_bytes=int(torch.mps.driver_allocated_memory()),
        host_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        parameter_hashes={k:rawsha(q.asarray(v)) for k,v in parameters.items()})


def verify_result(record,args,admitted,tick=None):
    require(record['schema']==SCHEMA and record['stage']=='train'
        and all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
            'proposal_complete','all_graphs_retired','all_parameter_and_optimizer_owners_retired',
            'all_inputs_and_normalizers_unchanged','all624_prefixes_exact','dead384_data_and_moments_zero',
            'terminal_only_payload_reloaded','all24_predictions_sealed_before_screen','no_native_gradient_claim'))
        and record['graphs_completed']==record['retired_call_count']==624 and record['training_graphs']==600
        and record['terminal_train_graphs']==20 and record['terminal_check_graphs']==4
        and record['updates']==record['parameter_updates']==150 and record['physical_model_calls']==0
        and record['training_targets_accessed'] and not record['validation_targets_accessed']
        and record['native_gradient_qualified'] is False and record['native_score_measured'] is False
        and record['prerequisite_seals']==admitted['prerequisite_seals'] and record['joint_inventory_sha256']==admitted['joint_sha256']
        and record['model_source_proof']==admitted['model_proof'] and record['cached_loader_proof']==admitted['loader_proof'],
        'One complete fixed proposal, no claimed native score or gradient')
    roster=check_roster(HERE/'training_v1/model_call_roster.json','train')
    require(len(record['graph_records'])==624 and len(record['update_records'])==150,'Exact all completed records')
    for spec,packet in zip(roster,record['graph_records']):
        require(all(packet[k]==v for k,v in spec.items()) and packet['graph_retired'] and packet['passed'] and packet['completed'],
            'Every original-order graph actually retired')
        p=HERE/'training_v1'/f"graph_{spec['graph_index']:04d}_complete.json"
        require(read(p)==packet and sha(p,tick)==record['artifacts'][str(p)],'Actual linked completed graph')
    for i,meta in enumerate(record['update_records']):
        p=HERE/'training_v1'/f'update_{i+1:03d}.json'; update=read(p)
        require(meta==dict(path=str(p),sha256=record['artifacts'][str(p)]) and sha(p,tick)==meta['sha256']
            and update['update']==i+1 and update['learning_rate']==1e-4+.5*(1e-3-1e-4)*(1+math.cos(math.pi*i/149))
            and math.isfinite(update['preclip_global_norm']) and update['preclip_global_norm']>=0
            and set(update['states'])==set(admitted['model'].PARAMETER_KEYS)
            and all(value['step']==i+1 for value in update['states'].values()) and update['dead384_exact_decay']
            and 0<=update['mps_driver_bytes']<DRIVER and 0<update['host_peak_rss_bytes']<RSS,
            'All150 actual fixed-rate/state/decay/resource update receipts')
    terminal=read(HERE/'training_v1/terminal_seal.json'); sealed=read(HERE/'training_v1/final_predictions_sealed.json')
    payload=record['terminal_payload']['arrays']; norms=record['normalizers']['arrays']
    require(set(payload)==set(admitted['model'].PARAMETER_KEYS) and all(v['dtype']=='float32'
        and v['shape']==list(admitted['model'].SHAPES[k]) and valid(v['raw_sha256']) for k,v in payload.items())
        and set(norms)=={'centers','scales','r'} and all(v['dtype']=='float32'
            and v['shape']==([] if k=='r' else [16,30]) and valid(v['raw_sha256']) for k,v in norms.items()),
        'Exact terminal FP32 ten-block payload and961 frozen normalizer/scale literals')
    require(terminal['updates']==150 and terminal['training_graphs']==600 and terminal['final_predictions_started'] is False
        and terminal['terminal_payload']==record['terminal_payload'] and terminal['normalizers']==record['normalizers']
        and sealed['terminal_seal_sha256']==record['terminal_seal_sha256'] and sealed['predictions']==record['final_predictions']
        and sealed['TRAIN_order']==list(TRAIN) and sealed['CHECK_order']==list(CHECK)
        and len(sealed['predictions'])==24 and sealed['validation_targets_accessed'] is False,
        'One terminal payload sealed before all target-blind final predictions')
    for path,digest in owners(args).items():
        require(record['source_bindings'].get(str(path))==digest,'Same reviewed training owners')
    require(record['target_bindings']=={v['path']:v['sha256'] for v in admitted['targets'].values()}
        and type(record['TRAIN_screen_passed']) is bool
        and record['TRAIN_screen_passed']==(record['TRAIN_screen']['ratio']<=.90),'Exact TRAIN-only eligibility screen, not operational gate')
    screen=record['TRAIN_screen']; rows=screen['rows']
    require([v['rank'] for v in rows]==list(TRAIN) and all(v['original_dtype']=='float32' for v in rows)
        and screen['projected_host_mean_mse']==math.fsum(v['projected_host_final_mse'] for v in rows)/20
        and screen['terminal_mean_mse']==math.fsum(v['final_mse'] for v in rows)/20
        and screen['projected_host_mean_mse']>0 and math.isfinite(screen['terminal_mean_mse'])
        and screen['ratio']==screen['terminal_mean_mse']/screen['projected_host_mean_mse']
        and screen['threshold']==.90 and screen['native_score'] is False,
        'Original-precision saved TRAIN scalar arithmetic, never a CHECK/native score')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    for key in FLAGS:
        parser.add_argument('--'+key+'-sha256')
    args=parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began=time.monotonic(); folder=HERE/'training_v1'
    require(sys.executable==AP and sys.dont_write_bytecode and sys.flags.optimize==0 and os.getpriority(os.PRIO_PROCESS,0)>=10
        and all(os.environ.get(k)=='1' for k in BLAS) and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
        and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO')=='0.1','Exact fixed AP/nice10/BLAS/MPS frame')
    require(not folder.exists() and not HERE.is_symlink(),'Exclusive new training stage'); folder.mkdir()
    result=dict(schema=SCHEMA,stage='train',passed=False,completed=False,source_bindings={},input_bindings={},
        weight_bindings={},target_bindings={},artifacts={},graph_records=[],update_records=[],graphs_completed=0,
        retired_call_count=0,updates=0,parameter_updates=0,physical_model_calls=0,training_graphs=0,
        terminal_train_graphs=0,terminal_check_graphs=0,training_targets_accessed=False,validation_targets_accessed=False,
        native_gradient_qualified=False,no_native_gradient_claim=True,native_score_measured=False,final_predictions={},
        fused_native_authorized=False,submission_authorized=False)
    torch=parameters=optimizer=reloaded=None
    def guard():
        require(time.monotonic()-began<CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<RSS,'Whole600s/host12GB')
        if torch is not None:
            require(torch.mps.driver_allocated_memory()<DRIVER,'Actual driver16GB')
    def json_artifact(name,value):
        p=folder/name; write(p,value); result['artifacts'][str(p)]=sha(p,guard)
        return dict(path=str(p),sha256=result['artifacts'][str(p)])
    def bundle(name,arrays):
        p=folder/name
        with p.open('xb') as f:
            np.savez(f,**arrays)
        digest=sha(p,guard); result['artifacts'][str(p)]=digest
        return dict(path=str(p),sha256=digest,arrays={k:arraymeta(v) for k,v in arrays.items()})
    def save_prediction(name,value):
        p=folder/name
        with p.open('xb') as f:
            np.save(f,value,allow_pickle=False)
        digest=sha(p,guard); result['artifacts'][str(p)]=digest
        return dict(path=str(p),sha256=digest,**arraymeta(value))
    old_handlers={s:signal.signal(s,lambda *unused: (_ for _ in ()).throw(RuntimeError('Training interrupted')))
                  for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        admitted=admit(args,result['source_bindings'],result['input_bindings'],result['weight_bindings'],guard)
        result.update(prerequisite_seals=admitted['prerequisite_seals'],joint_inventory_sha256=admitted['joint_sha256'],
            model_source_proof=admitted['model_proof'],cached_loader_proof=admitted['loader_proof'],
            physical_candidate_sha256=admitted['candidate_sha256'])
        require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),'All actual eligibility precedes labels and numerics')
        import numpy as np
        q=admitted['q']; model=admitted['model']; torch=model.runtime()
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(torch.get_num_threads()==torch.get_num_interop_threads()==1 and torch.backends.mps.is_available(), 'Exact Torch1/MPS')
        require(Path(np.__file__).resolve()==SITE/'numpy/__init__.py' and Path(torch.__file__).resolve()==SITE/'torch/__init__.py','Actual admitted module origins')
        rng=np.random.Generator(np.random.PCG64DXSM(93091230)); orders=[rng.permutation(TRAIN).tolist() for _ in range(30)]; del rng
        roster=entries(orders); json_artifact('model_call_roster.json',dict(schema='current_v29_temporal_observer_training_roster_v1',
            seed=93091230,generator='PCG64DXSM',orders=orders,entries=roster))
        centers,scales=q.train_normalizers(np,admitted['train'])
        with np.load(HERE/'neural_qualification_v2/train_feature_normalizers.npz',allow_pickle=False) as fixed:
            require(centers.tobytes()==fixed['centers'].tobytes() and scales.tobytes()==fixed['scales'].tobytes(),'Same qualified TRAIN-only normalizers')
        targets,r,baseline=targets_and_scale(np,admitted,result,guard)
        normalizers=dict(centers=centers,scales=scales,r=r); result['normalizers']=bundle('normalizers.npz',normalizers)
        frozen_normalizers={k:rawsha(v) for k,v in normalizers.items()}
        values=q.initial_values(np,model); parameters=model.initialize('mps',trainable=True)
        require(all(q.asarray(parameters[k]).tobytes()==v.tobytes() for k,v in values.items()),'Fresh exact initialization, no old payload')
        result['initial_parameter_hashes']={k:rawsha(v) for k,v in values.items()}; del values
        optimizer=torch.optim.AdamW(list(parameters.values()),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=2e-5,foreach=False)
        for spec in roster:
            index=spec['graph_index']; guard()
            if index<600 and index%4==0:
                update=index//4; lr=1e-4+.5*(1e-3-1e-4)*(1+math.cos(math.pi*update/149))
                optimizer.param_groups[0]['lr']=lr; optimizer.zero_grad(set_to_none=True)
            if index==600:
                payload={k:q.asarray(v) for k,v in parameters.items()}
                result['terminal_payload']=bundle('terminal_payload.npz',payload)
                seal=json_artifact('terminal_seal.json',dict(passed=True,completed=True,updates=150,training_graphs=600,
                    final_predictions_started=False,terminal_payload=result['terminal_payload'],normalizers=result['normalizers'],
                    source_bindings=result['source_bindings'],orders_sha256=result['artifacts'][str(folder/'model_call_roster.json')]))
                result['terminal_seal_sha256']=seal['sha256']
                refs=[weakref.ref(optimizer)]+[weakref.ref(v) for v in parameters.values()]
                refs.extend(weakref.ref(v) for v in (p.grad for p in parameters.values()) if v is not None)
                refs.extend(weakref.ref(v) for state in optimizer.state.values() for v in state.values())
                parameters=optimizer=targets=None; gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()
                require(all(v() is None for v in refs),'All train parameters/optimizer/gradients retired before terminal reload')
                with np.load(result['terminal_payload']['path'],allow_pickle=False) as fixed:
                    loaded={k:fixed[k].copy() for k in fixed.files}
                require(set(loaded)==set(model.PARAMETER_KEYS) and all(loaded[k].tobytes()==payload[k].tobytes() for k in loaded),'Only sealed terminal payload reloaded')
                reloaded=q.tensor_parameters(torch,loaded,'mps',torch.float32,False); del payload,loaded
            cap=admitted['check'] if spec['kind']=='terminal_check_forward' else admitted['train']
            source=admitted['loader'](np,cap,spec['rank'],centers,scales); source['r']=r
            packet,served,refs=run_one(np,torch,q,model,spec,source,parameters if index<600 else reloaded,
                target=targets[spec['rank']] if index<600 else None)
            del source; gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache(); torch.mps.synchronize()
            require(all(v() is None for v in refs),'Every network graph/input/power/state actually retired')
            require(frozen_normalizers=={k:rawsha(v) for k,v in normalizers.items()},'All normalizer/r bytes fixed')
            fixture=cap['fixture_bindings'][str(spec['rank'])]
            packet['canonical_weight_hashes']=dict(payload_sha256=packet['input_hashes']['weights'],
                STORE_dtype_shape_payload_sha256=fixture['raw_sha256'],file_sha256=fixture['sha256'])
            if index>=600:
                result['final_predictions'][str(spec['rank'])]=save_prediction(f'final_rank{spec["rank"]:04d}.npy',served)
            del served
            packet.update(passed=True,completed=True,graph_retired=True,host_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                mps_driver_bytes=int(torch.mps.driver_allocated_memory()),elapsed_s=time.monotonic()-began)
            guard(); meta=json_artifact(f'graph_{index:04d}_complete.json',packet)
            result['graph_records'].append(packet); result['graphs_completed']+=1; result['retired_call_count']+=1
            result['training_graphs' if index<600 else 'terminal_train_graphs' if index<620 else 'terminal_check_graphs']+=1
            print(json.dumps(dict(event='retired',call=index,rank=spec['rank'],kind=spec['kind'],device='mps',
                complete_sha256=meta['sha256'],host_peak_rss_bytes=packet['host_peak_rss_bytes'],mps_driver_bytes=packet['mps_driver_bytes'])),flush=True)
            if index<600 and index%4==3:
                update_packet=update_once(np,torch,q,optimizer,parameters,index//4,lr); guard()
                result['update_records'].append(json_artifact(f'update_{index//4+1:03d}.json',update_packet))
                result['updates']+=1; result['parameter_updates']+=1
        refs=[weakref.ref(v) for v in reloaded.values()]; reloaded=None; gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()
        require(all(v() is None for v in refs),'Terminal parameters also retired')
        seal=json_artifact('final_predictions_sealed.json',dict(passed=True,completed=True,
            terminal_seal_sha256=result['terminal_seal_sha256'],TRAIN_order=list(TRAIN),CHECK_order=list(CHECK),
            predictions=result['final_predictions'],validation_targets_accessed=False,screen_evaluated=False))
        result['final_predictions_seal_sha256']=seal['sha256']; train_rows=[]
        for rank in TRAIN:
            info=admitted['targets'][rank]; hash_target(info['path'],info['sha256'],admitted['targets'],guard)
            with np.load(info['path'],allow_pickle=False) as fixed:
                target=fixed['mc_post'].copy()
            prediction=np.load(result['final_predictions'][str(rank)]['path'],allow_pickle=False)
            raw=float(np.mean((prediction[15].astype(np.float64)-target[15].astype(np.float64))**2))
            train_rows.append(dict(rank=rank,final_mse=raw,**baseline[str(rank)]))
        initial=math.fsum(row['projected_host_final_mse'] for row in train_rows)/20
        final=math.fsum(row['final_mse'] for row in train_rows)/20
        require(initial>0 and math.isfinite(final),'Finite original-precision TRAIN screen')
        result['TRAIN_screen']=dict(rows=train_rows,projected_host_mean_mse=initial,terminal_mean_mse=final,ratio=final/initial,
            raw_host_mean_mse=math.fsum(row['raw_host_final_mse'] for row in train_rows)/20,threshold=.90,
            native_score=False,baseline_semantics='actual_projected_head_free_served')
        result['TRAIN_screen_passed']=final/initial<=.90
        for group in ('source_bindings','input_bindings','weight_bindings','artifacts'):
            for path,digest in result[group].items():
                require(sha(path,guard)==digest,'Final complete nontarget/source/artifact bytes unchanged')
        for path,digest in result['target_bindings'].items():
            hash_target(path,digest,admitted['targets'],guard)
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,proposal_complete=True,
            all_graphs_retired=True,all_parameter_and_optimizer_owners_retired=True,all_inputs_and_normalizers_unchanged=True,
            all624_prefixes_exact=True,dead384_data_and_moments_zero=True,terminal_only_payload_reloaded=True,
            all24_predictions_sealed_before_screen=True,model_parameters=44209,deployed_literals=45170)
        guard()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        parameters=optimizer=reloaded=None; gc.collect()
        if torch is not None:
            torch.mps.synchronize(); torch.mps.empty_cache()
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if result['elapsed_s']>=CAP or result['peak_rss_bytes']>=RSS:
        result.update(passed=False,completed=False)
    write(folder/'result.json',result); rh=sha(folder/'result.json'); ok=bool(result['passed'] and time.monotonic()-began<CAP)
    write(folder/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch=sha(folder/'completion.json')
    elapsed=time.monotonic()-began; ok=bool(ok and elapsed<CAP)
    write(folder/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
    sha(folder/'finalization.json'); ok=bool(ok and time.monotonic()-began<CAP)
    print(json.dumps(dict(passed=ok,completed=ok,retired_call_count=result['retired_call_count'],error=result.get('error'))),flush=True)
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())
