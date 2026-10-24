"""One fixed anchored TRAIN20 proposal; numerical APIs/owners are injected.

No numerical imports, files, stages, targets or fitting at import. Admission
and the external1500s/8GB/zero-swap supervisor are separate owners. evaluate
implements exactly20 baseline,20 checkpoint/backward,20 terminal calls and
five Adam updates. Only detached bytes/plain metadata leave this engine.
The native anchor is used ONLY at the terminal loss/report boundary.
"""
import gc
import hashlib
import io
import math
import os
from pathlib import Path
import sys
import time
import types
import weakref

SCHEMA = 'current_shared384_anchored_p_train_engine_v1'
TRAIN = (35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
COST_UPPER = 134_939_477_450_240


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path, tick):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'Regular admitted file')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick is not None: tick()
    return h.hexdigest()


def raw_bytes(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


def tensor_hash(value):
    return digest(raw_bytes(value))


def remember(torch, refs, value):
    if isinstance(value,torch.Tensor): refs.append(weakref.ref(value))
    elif isinstance(value,dict):
        for item in value.values(): remember(torch,refs,item)
    elif isinstance(value,(tuple,list)):
        for item in value: remember(torch,refs,item)


def remember_numpy(np, refs, value):
    """Track view AND actual base; do not retain either in the observer."""
    while isinstance(value,np.ndarray):
        refs.append(weakref.ref(value)); value=value.base


def owned_weights(np, row, refs, tick):
    require(file_sha(row['weights'],tick)==row['weights_sha256'],'Same canonical physical weight file')
    loaded=np.load(row['weights'],allow_pickle=False); loader_refs=[]
    remember_numpy(np,loader_refs,loaded)
    require(type(loaded) is np.ndarray and loaded.shape==(16,1024,1024)
            and loaded.dtype==np.float32 and loaded.flags.c_contiguous,'Canonical typed weight ingress')
    before=digest(loaded.tobytes()); weights=loaded.copy(order='K')
    require(weights.flags.owndata and weights.flags.c_contiguous and weights.base is None
            and digest(weights.tobytes())==before and bool(np.isfinite(weights).all()),
            'Qualified explicit owned byte-preserving finite weight copy')
    del loaded; gc.collect()
    require(all(ref() is None for ref in loader_refs),'Loader view/base retired BEFORE the model')
    remember_numpy(np,refs,weights)
    return weights,before


def tensor_from_bytes(np, torch, raw, refs):
    require(type(raw) is bytes and len(raw)==65536,'Only complete detached all16 FP32 bytes')
    view=np.frombuffer(raw,dtype=np.float32).reshape(16,1024)
    remember_numpy(np,refs,view)
    result=torch.tensor(view,dtype=torch.float32,device='cpu')
    remember(torch,refs,result)
    return result


def file_tensor(np, torch, path, expected, key, refs, tick):
    require(file_sha(path,tick)==expected,'Exact admitted numerical file immediately before decode')
    if key is None:
        value=np.load(path,allow_pickle=False)
    else:
        with np.load(path,allow_pickle=False) as archive:
            require(key in archive.files,'Only declared original target member')
            value=archive[key]
    remember_numpy(np,refs,value)
    require(type(value) is np.ndarray and value.shape==(16,1024) and value.dtype==np.float32
            and bool(np.isfinite(value).all()),'Original finite stored FP32 all16 frame')
    raw=value.tobytes()
    result=torch.tensor(value,dtype=torch.float32,device='cpu')
    remember(torch,refs,result)
    return result,raw


def clear_constant_caches(model, torch, refs):
    co=model._JOINT._CO
    if co._PAYLOAD is not None:
        refs.extend(weakref.ref(value) for value in co._PAYLOAD.values())
        co._PAYLOAD=None
    cdf=sys.modules.get('gaussian_cdf_quadrature_v1')
    require(isinstance(cdf,types.ModuleType),'Actual qualified CDF constant owner')
    remember(torch,refs,cdf._CONSTANTS); cdf._CONSTANTS.clear()


def evaluate(prepared, np, torch, workspace, attestation,
             on_call, on_update, on_payload, tick=None):
    """Return (plain result, filename->bytes); callbacks do not receive arrays.

    Calls0..19 baseline,20..39 training,40..59 terminal. on_call occurs ONLY
    after every card/graph/input/loader owner has retired. P/A, independent
    accumulators and Adam are explicitly persistent, separately retired later.
    on_update(index0..4,record) contains hashes only. on_payload(record,bytes)
    is called exactly once after five updates and before terminal reload.
    Caller retains responsibility for workspace/backend retirement and final
    source/input/TRAIN-truth rehashes under its independently owned watch.
    """
    started=time.monotonic(); model=prepared['model']; rows=prepared['rows']
    require(prepared['schema']=='current_shared384_anchored_p_train_admission_v1'
            and prepared['train']==list(TRAIN) and [r['rank'] for r in rows]==list(TRAIN)
            and prepared['cost_certificate']['complete_source_upper']==COST_UPPER
            and prepared['whole_seconds']==1500. and prepared['rss_bytes']==8_000_000_000,
            'Exact selected admission/roster/source work/resource envelope')
    require(callable(on_call) and callable(on_update) and callable(on_payload),'Explicit owned publication callbacks')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in BLAS)
            and torch.get_num_threads()==torch.get_num_interop_threads()==1
            and torch.is_grad_enabled() and not {'flopscope','scipy'}&set(sys.modules),
            'Qualified CPU backend, all threads1, no hidden native/MPS stage')
    keys=tuple(model._JOINT.PKEYS); inactive=tuple(model._JOINT.INACTIVE_KEYS)
    require(len(keys)==11,'All eleven existing P parameter blocks')
    P,A=model.frozen('cpu',trainable_P=True)
    require(sum(P[k].numel() for k in keys)==45200 and not A.requires_grad
            and all(v.is_leaf and v.requires_grad==(k in keys) for k,v in P.items()),
            'Exact existing trainable leaves, frozen NODE A')
    initial={k:tensor_hash(v) for k,v in P.items()}
    initial_inactive={k:digest(raw_bytes(P[k][0])) for k in inactive}
    shapes={k:list(v.shape) for k,v in P.items()}; A_hash=tensor_hash(A)
    expected_cstar=model._JOINT._CO.coefficient_bytes(model._JOINT._LITERALS['_BD_COEFFICIENT_LITERAL'])
    cstar_hash=digest(expected_cstar)
    persistent_refs=[weakref.ref(v) for v in P.values()]+[weakref.ref(A)]
    params=tuple(P[k] for k in keys)
    accumulators={k:torch.zeros_like(P[k],requires_grad=False) for k in keys}
    remember(torch,persistent_refs,accumulators)
    require(len({v.data_ptr() for v in accumulators.values()})==11 and all(
        accumulators[k].data_ptr()!=P[k].data_ptr() for k in keys),'Independent gradient accumulation slots')
    optimizer=torch.optim.Adam(params,lr=5e-4,betas=(.9,.999),eps=1e-8,
        weight_decay=0.,foreach=False,fused=False)
    persistent_refs.append(weakref.ref(optimizer))
    # Scalar cost weights are research arithmetic, not a prediction of new F.
    own=torch.tensor([r['own_cold_flops'] for r in rows],dtype=torch.float64)
    multipliers=torch.maximum(own/(2**41),torch.full_like(own,.1))
    cost_weights=multipliers/multipliers.mean()
    remember(torch,persistent_refs,(own,multipliers,cost_weights)); del own,multipliers
    require(bool(torch.isfinite(cost_weights).all()) and bool((cost_weights>0).all()),'Finite fixed own-cost weights')
    weight_values=cost_weights.tolist()  # small plain scalar reporting only
    baseline_bytes={}; records=[]; updates=[]; artifacts={}; truths={}; payload_record=None
    expected_proof=prepared['prepared_qualification']['shape_control']['shared_source_proof']

    def guard_frozen(parameters, gradients_none=True):
        require(set(parameters)==set(initial) and tensor_hash(A)==A_hash and A.grad is None,
                'Complete P surface and unchanged frozen A')
        require(all(v.dtype==torch.float32 and v.device.type=='cpu' and list(v.shape)==shapes[k]
                and bool(torch.isfinite(v).all()) for k,v in parameters.items()),'Finite exact P layouts')
        require(all(tensor_hash(parameters[k])==initial[k] for k in initial if k not in keys)
                and all(digest(raw_bytes(parameters[k][0]))==initial_inactive[k] for k in inactive),
                'All non-P/normalizer/input fields and inactive66 bytes unchanged')
        if gradients_none:
            require(all(v.grad is None for v in parameters.values()),'No persistent leaf .grad during card execution')

    def compute_call(call, parameters):
        ordinal=call%20; row=rows[ordinal]; phase=('baseline','training','terminal')[call//20]
        training=phase=='training'; began=time.monotonic(); refs=[]; input_refs=[]
        guard_frozen(parameters); before={k:tensor_hash(v) for k,v in parameters.items()}
        weights,weight_raw=owned_weights(np,row,input_refs,tick)
        native,native_bytes=file_tensor(np,torch,row['native_output'],row['native_output_sha256'],
                                       None,input_refs,tick)
        require(digest(native_bytes)==row['native_output_raw_sha256'] and bool((native>=0).all()),
                'Actual all16 served native anchor bytes/domain')
        target=None
        if phase!='baseline':
            require(len(baseline_bytes)==20 and row['target_sha256']==prepared['truth_bindings'][row['target']],
                    'All baseline cards sealed before the fixed TRAIN label is opened')
            target,target_bytes=file_tensor(np,torch,row['target'],row['target_sha256'],
                                           row['target_key'],input_refs,tick)
            truths[row['target']]=row['target_sha256']
            del target_bytes
        card=model.card(row['rank'],P=parameters,A=A,device='cpu',weights=weights,
            native_workspace=workspace,native_attestation=attestation,numpy_api=np)
        refs.extend((weakref.ref(card),weakref.ref(card.coefficients),weakref.ref(card._native_rangefinder)))
        remember(torch,refs,(card.w,card.ws,card.fixed))
        require(raw_bytes(card.coefficients)==expected_cstar and not card.coefficients.requires_grad
                and not card.w.requires_grad and not card.ws.requires_grad and not card.fixed.requires_grad,
                'Exact frozen physical coefficients/weights/features')
        source_step,source_select,source_shared=card.step,card._select,card._record_shared
        def step(*args):
            remember(torch,refs,args[1:3])  # state/control only, never persistent P
            result=source_step(*args); remember(torch,refs,result)
            if tick is not None: tick()
            return result
        def select(score):
            indices,mask=source_select(score)
            require(0<len(indices)<=1024 and mask.shape==(1024,),'Source-priced legal fresh selection')
            remember(torch,refs,(score,indices,mask)); return indices,mask
        def shared(layer,pp,U,F,core,old,stops):
            source_shared(layer,pp,U,F,core,old,stops)
            widths=[b-a for a,b in zip((0,)+tuple(stops),stops)]
            require(pp.shape[0]==1024 and len(stops)==layer and all(0<w<=2048 and w%2==0 for w in widths)
                    and pp.shape[1]<=2048*layer and old<=20480,'Within complete max-geometry source certificate')
            if layer>=5:
                require(U.shape==(1024,384) and F.shape==(384,old) and core.shape==(384,384),
                        'Original rank384 shared state only')
            remember(torch,refs,(pp,U,F,core))
        card.step,card._select,card._record_shared=step,select,shared
        if training:
            raw,trace=card.run(training=True); served=model._JOINT._CO.served(raw)
        else:
            with torch.no_grad():
                raw,trace=card.run(training=False); served=model._JOINT._CO.served(raw)
        remember(torch,refs,(raw,trace,served))
        require(served.shape==(16,1024) and served.dtype==torch.float32
                and bool(torch.isfinite(served).all()) and bool((served>=0).all()),'All16 finite served surrogate')
        output=raw_bytes(served)
        cached=output if phase=='baseline' else baseline_bytes[row['rank']]
        baseline=tensor_from_bytes(np,torch,cached,input_refs)
        delta=served-baseline                     # FP32 subtraction FIRST
        anchored=model._JOINT._CO.served(native+delta)  # FP32 add/serve, LOSS ONLY
        remember(torch,refs,(delta,anchored))
        require(delta.dtype==anchored.dtype==torch.float32 and bool(torch.isfinite(anchored).all())
                and bool((anchored>=0).all()),'Exact selected FP32 anchored circuit')
        anchored_bytes=raw_bytes(anchored)
        at_P0=phase=='baseline' or (training and ordinal<4)
        if at_P0:
            require(all(before[k]==initial[k] for k in keys) and anchored_bytes==native_bytes,
                    'P0 anchored served output byte-exact to native, including first checkpoint batch')
        metrics={}; gradient_meta={}
        if target is not None:
            error=anchored.to(torch.float64)-target.to(torch.float64)
            final=(error[15]*error[15]).mean(); earlier=(error[:15]*error[:15]).mean()
            unweighted=(final+.05*earlier)/2e-8
            weight=cost_weights[ordinal]
            loss=weight*unweighted/4.
            unanchored_error=served.detach().to(torch.float64)-target.to(torch.float64)
            metrics=dict(anchored_final_mse=float(final.detach()),anchored_earlier_mse=float(earlier.detach()),
                unweighted_loss=float(unweighted.detach()),weighted_loss=float((weight*unweighted).detach()),
                unanchored_final_mse=float((unanchored_error[15]*unanchored_error[15]).mean()),
                normalized_cost_weight=weight_values[ordinal])
            require(all(math.isfinite(v) for v in metrics.values()) and bool(torch.isfinite(loss)),
                    'Finite original-label FP64 loss/reporting')
            remember(torch,refs,(error,final,earlier,unweighted,loss,unanchored_error))
            if training:
                gradients=torch.autograd.grad(loss,tuple(parameters[k] for k in keys),
                    allow_unused=False,create_graph=False,retain_graph=False)
                remember(torch,refs,gradients)
                with torch.no_grad():
                    for key,grad in zip(keys,gradients):
                        require(grad.shape==parameters[key].shape and grad.dtype==torch.float32
                                and not grad.requires_grad and bool(torch.isfinite(grad).all()),
                                'Every connected finite complete P gradient')
                        if key in inactive:
                            require(bool((grad[0]==0).all()),'All inactive66 per-card gradients exactly zero')
                        accumulators[key].add_(grad)
                        wide=grad.to(torch.float64)
                        gradient_meta[key]=dict(norm=float(torch.sqrt((wide*wide).sum())),
                            sha256=tensor_hash(grad),shape=list(grad.shape),connected=True)
                        require(math.isfinite(gradient_meta[key]['norm'])
                                and bool(torch.isfinite(accumulators[key]).all()),'Finite independent gradient sum')
                del gradients,grad,wide
        diagnostic=model._JOINT._CO.check_replay(card)
        require(set(diagnostic['shared_cells'])=={str(l) for l in range(16)}
                and set(diagnostic['node_head_calls'])=={str(l) for l in range(1,15)}
                and diagnostic['shared_source_proof']==expected_proof,'Exact original full shared/NODE/source ownership')
        for name in ('selection_visits','variance_visits','shared_visits','node_head_calls'):
            require(all(1<=v<=(2 if training else 1) for v in diagnostic[name].values()),
                    'No missing or unpriced repeated card work: '+name)
        require(all(tensor_hash(v)==before[k] and v.grad is None for k,v in parameters.items())
                and raw_bytes(card.coefficients)==expected_cstar and digest(weights.tobytes())==weight_raw,
                'No parameter/physical input mutation during a card')
        guard_frozen(parameters)
        del card.step,card._select,card._record_shared
        produced={f'call_{call:02d}_served.f32':output}
        if phase!='baseline': produced[f'call_{call:02d}_anchored.f32']=anchored_bytes
        record=dict(call=call,rank=row['rank'],stage=phase,backward=training,truth_accessed=phase!='baseline',
            model_elapsed_s=time.monotonic()-began,served_sha256=digest(output),
            anchored_sha256=digest(anchored_bytes),native_anchor_sha256=digest(native_bytes),
            baseline_sha256=digest(cached),P0_anchor_check_applicable=at_P0,
            P0_anchor_byte_exact=True if at_P0 else None,
            weight_file_sha256=row['weights_sha256'],own_cold_flops=row['own_cold_flops'],
            metrics=metrics,gradient_blocks=gradient_meta,diagnostics=diagnostic,
            parameters_unchanged_within_call=True,frozen_A_cstar_nonP_inactive_unchanged=True,
            persistent_parameters_accumulators_optimizer_excluded_from_per_card_retirement=True)
        return record,produced,refs,input_refs

    def run_call(call, parameters):
        record,produced,refs,input_refs=compute_call(call,parameters)
        gc.collect()
        require(all(ref() is None for ref in refs),'Every actual card/graph/output owner retired')
        require(all(ref() is None for ref in input_refs),'Every weight/loader/native/target/baseline input owner retired')
        record.update(call_graph_owners_retired=True,input_owners_retired=True)
        if call<20: baseline_bytes[TRAIN[call]]=produced[f'call_{call:02d}_served.f32']
        artifacts.update(produced); records.append(record)
        on_call(call,record,produced)
        require(time.monotonic()-started<1500,'Whole engine within independently watched1500s')
        if tick is not None: tick()

    for call in range(20): run_call(call,P)
    require(len(baseline_bytes)==20 and not truths and all(tensor_hash(v)==initial[k] for k,v in P.items()),
            'Exactly20 original-P0 baselines precede any target decode or update')
    for update in range(5):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for value in accumulators.values(): value.zero_()
        del value
        for ordinal in range(4*update,4*update+4): run_call(20+ordinal,P)
        lr=5e-5+.5*(5e-4-5e-5)*(1+math.cos(math.pi*update/4))
        optimizer.param_groups[0]['lr']=lr
        step_refs=[]
        for key in keys:
            # Adam receives independent short-lived .grad copies; its writes/
            # clipping cannot alias per-card gradients or persistent sums.
            P[key].grad=accumulators[key].clone(); remember(torch,step_refs,P[key].grad)
            require(P[key].grad.data_ptr()!=accumulators[key].data_ptr(),'Independent optimizer gradient slot')
        norm=torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True,foreach=False)
        norm_value=float(norm); remember(torch,step_refs,norm); del norm
        require(math.isfinite(norm_value),'Finite global pre-clip norm')
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        guard_frozen(P); states={}
        require(len(optimizer.state)==11,'Exactly eleven actual Adam states')
        for key in keys:
            state=optimizer.state[P[key]]
            require(set(state)=={'step','exp_avg','exp_avg_sq'},'Fixed ordinary Adam state, no extra algorithm')
            step_value=float(state['step'])
            require(math.isfinite(step_value) and step_value==update+1,'Every exact Adam step index')
            for name in ('exp_avg','exp_avg_sq'):
                require(state[name].shape==P[key].shape and state[name].dtype==torch.float32
                        and not state[name].requires_grad and bool(torch.isfinite(state[name]).all()),'Finite complete Adam moment')
                if key in inactive: require(bool((state[name][0]==0).all()),'Inactive Adam moments stay zero')
            states[key]=dict(step=int(step_value),exp_avg_sha256=tensor_hash(state['exp_avg']),
                            exp_avg_sq_sha256=tensor_hash(state['exp_avg_sq']))
        del state
        gc.collect(); require(all(ref() is None for ref in step_refs),'Optimizer .grad/clip temporary owners retired')
        record=dict(update=update,optimizer_step=update+1,ranks=list(TRAIN[4*update:4*update+4]),
            learning_rate=lr,pre_clip_gradient_norm=norm_value,adam_states=states,
            parameter_hashes={k:tensor_hash(v) for k,v in P.items()},
            frozen_A_cstar_nonP_inactive_unchanged=True,inactive66_gradient_moments_zero=True,
            checkpoint_saved=False,parameter_updates=update+1)
        updates.append(record); on_update(update,record)
        if tick is not None: tick()
    require(len(records)==40 and len(updates)==5 and set(truths)==set(prepared['truth_bindings']),
            'Exactly20 training calls/five updates/all and only20 TRAIN labels')

    def seal_payload():
        refs=[]; arrays={}
        for key in sorted(P):
            arrays[key]=np.array(P[key].detach().numpy(),dtype=np.float32,copy=True,order='C')
            remember_numpy(np,refs,arrays[key])
            require(digest(arrays[key].tobytes())==tensor_hash(P[key]),'Exact terminal tensor export')
        stream=io.BytesIO(); np.savez(stream,**arrays); blob=stream.getvalue(); stream.close()
        metadata=dict(sha256=digest(blob),nbytes=len(blob),terminal_epoch=1,parameter_updates=5,
            training_graphs=20,trainable_keys=list(keys),shapes=shapes,dtype='float32',
            parameter_hashes={k:tensor_hash(v) for k,v in P.items()},A_sha256=A_hash,cstar_sha256=cstar_hash,
            original_candidate_sha256=prepared['candidate_sha256'],checkpoint_selection=False,
            validation_targets_accessed=False,native_score_measured=False,native_gradient_certified=False)
        return blob,metadata,refs
    payload,payload_record,payload_refs=seal_payload()
    gc.collect(); require(all(ref() is None for ref in payload_refs),'Temporary terminal NPZ array owners retired')
    artifacts['terminal_payload.npz']=payload
    on_payload(payload_record,payload)
    # Retire the entire trained leaf/Adam/accumulator lifetime before reload.
    remember(torch,persistent_refs,optimizer.state)
    optimizer.zero_grad(set_to_none=True); optimizer.state.clear()
    optimizer.param_groups.clear(); del optimizer,params,accumulators
    P.clear(); del P
    gc.collect()
    # A/cost_weights intentionally remain for the final no-grad cards; only the
    # old P/Adam state/accumulation subset must already be gone here.
    survivors=[ref() for ref in persistent_refs if ref() is not None]
    require(len(survivors)==2 and all(v is A or v is cost_weights for v in survivors),
            'Only explicit frozen A and scalar cost weights survive trained-owner retirement')
    del survivors

    def reload_terminal():
        refs=[]
        require(digest(payload)==payload_record['sha256'],'Sole sealed terminal bytes before reload')
        with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
            require(set(archive.files)==set(initial),'Full sole terminal P member roster')
            values={k:archive[k] for k in archive.files}
        for value in values.values(): remember_numpy(np,refs,value)
        result={k:torch.tensor(value,dtype=torch.float32,device='cpu',requires_grad=False)
                for k,value in values.items()}
        require(all(list(v.shape)==shapes[k] and tensor_hash(v)==payload_record['parameter_hashes'][k]
                for k,v in result.items()),'Every reloaded terminal P tensor matches its sole seal')
        return result,refs
    terminal,decode_refs=reload_terminal()
    gc.collect(); require(all(ref() is None for ref in decode_refs),'Temporary payload loader/base owners retired')
    remember(torch,persistent_refs,terminal); guard_frozen(terminal)
    for call in range(40,60): run_call(call,terminal)
    require(len(records)==60 and len(updates)==5,'No extra call/update/epoch/terminal selection')
    require(all(tensor_hash(v)==payload_record['parameter_hashes'][k] for k,v in terminal.items()),
            'Reloaded sole terminal unchanged across all20 diagnostic forwards')
    terminal.clear(); del terminal,A,cost_weights
    clear_constant_caches(model,torch,persistent_refs)
    gc.collect(); require(all(ref() is None for ref in persistent_refs),'All engine numerical/parameter/optimizer/cache owners retired')
    require(time.monotonic()-started<1500,'Whole selected engine duration')
    return dict(schema=SCHEMA,passed=True,completed=True,roster=list(TRAIN),records=records,updates=updates,
        full_model_calls=60,baseline_forwards=20,training_forwards=20,backwards=20,terminal_forwards=20,
        parameter_updates=5,epochs=1,terminal_payload=payload_record,truth_accessed=True,
        truth_bindings=truths,validation_targets_accessed=False,baseline20_before_truth=True,
        checkpoint_selection=False,native_score_measured=False,native_gradient_certified=False,
        old_failed_gates_preserved=True,anchors_used_only_at_loss=True,teacher_states_used=False,
        all_engine_numerical_owners_retired=True,caller_workspace_backend_retirement_required=True,
        source_cost_upper=COST_UPPER,actual_flopscope_bill_claimed=False,
        normalized_cost_weights=weight_values,elapsed_s=time.monotonic()-started,
        automatic_next_stage=False),artifacts
