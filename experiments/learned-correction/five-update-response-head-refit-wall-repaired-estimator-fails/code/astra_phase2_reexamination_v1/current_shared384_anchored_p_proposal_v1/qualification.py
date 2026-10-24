"""Inert execution-integrity helper for ONE approximate current-shared proposal.

No numerical imports, fixture decoding, stage, optimizer or fit at import.
prepare is stdlib-only. The owner selects/attests the already pinned backend,
loads the admitted weights and allocates one explicit native workspace.
evaluate owns three fresh cards and their P/A, cotangent and gradient arrays.
Only detached bytes and plain metadata escape. Caller-owned input/workspace
retirement, whole120s/tree8GB/zero-swap monitoring and linked seals are external.

This is NOT a continuation/pass of the failed native-fidelity or NADJ tests.
No native output, teacher state, stored H, label, secant or optimizer is used.
The only prospective question is same-surrogate full-causal checkpoint/repeat
execution, with every current P/NODE/shared-state path preserved.
"""
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types
import weakref

HERE = Path(__file__).resolve().parent
CAUSAL = HERE.parent/'current_shared384_causal_p_v1'
P2 = HERE.parent.parent
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
WEIGHTS = STORE/'rows/observer_fit/0035/weights.npy'
C2_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
ALIGNMENT_SHA = '16cac20951f5a39078e18a641bd894518cae9c3b112dd922fa87540cb4a6af83'
OPERATOR_SHA = '09136ff645869df9d8fae3e2e8c16ff65c777971c308d1ba8c124d35649d2ad2'
SCHEMA = 'current_shared384_anchored_p_execution_v1'
ROSTER = ('baseline35','checkpoint35','checkpoint35_repeat')
LIMITS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
WORK_LIMIT = 8_000_000_000_000
NEGATIVE = ('cb7734bea7845ded9a887d0f92a76483390174b1747bed1bafe588fd597aea83',
 'dc34ed29a28d58f37eb964ba3185884773e1f5989b724bbcef9ddfa3bb228471',
 'db7441e5f443a3eb4a5d9dbd928920e2ace7e9e11cb1ee61a5966281641141f8')
NEGATIVE_WATCH = ('57652b969afb769a1602c82bff2655b172f630ab9b123aeaf6433ef6db278fb5',
 '40b87179911b9b6f42fd1245cd52e25954f0211de4b28c3e6f280a599403a4da',
 '128686a3dfeb261ea00e17a91bdad88b9abe519d0aec9109524c501bb3ea9f42')
VJP = ('55eec7f39a7c17af1e65c89b72d22b38b22bb40508db9e2acc6297ff2610e805',
 '5502f31a7aafc96ccb4f854d44875684aa24ffd109cb69195321a12bf904f368',
 'd390c453399aa868ecb0d3fbeeb08a2837a7e5cd5658f65a2f913c56724246be')
VJP_WATCH = ('5b5a54d3aa9f2aeb3c7db579dd4a7aff8323070389daef859e47bbc482148c88',
 '236b03f64ecc83ef4f6d451b956224a7f95385cc0ed941b45098de3ec97e0991',
 '771d542c449ce3c695929916dbe61c0c2db0f8e2f6c3dc5d4ba175695f1e881c')
DIAGNOSTIC_SHA = '206c33a4142f103dae58389139c2ef7fe1c81a4542e5db43a4b2179ed99e07e7'


def require(ok,message):
    if not ok: raise RuntimeError(message)


def sha(path,tick=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick is not None: tick()
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _seals(folder,digests,passed,bindings,tick):
    paths=[folder/name for name in ('result.json','completion.json','finalization.json')]
    for path,digest in zip(paths,digests):
        require(sha(path,tick)==digest,'Actual immutable prerequisite seal')
        bindings[str(path)]=digest
    r,c,f=map(_read,paths)
    require(all(x['passed'] is passed and x['completed'] is True for x in (r,c,f))
            and c['result_sha256']==f['result_sha256']==digests[0]
            and f['completion_sha256']==digests[1],'Preserved actual three-seal outcome')
    return r


def _join(watch,digests):
    require(watch['whole_owned_tree_retired'] and watch['maximum_swap_growth_bytes']==0
            and all(watch['child_'+k+'_sha256']==v for k,v in
                    zip(('result','completion','finalization'),digests)),
            'Actual child/watch join and retired zero-swap history')


def work_certificate(diagnostics):
    """Explicit scalar source bound; no tensor arithmetic or operation replay.

    Only the fixed current35 census is supported. Counts are checked before
    selected operations in each fresh run; every checkpoint visit is <=2.
    Matrix upper uses 2*m*k*n and rate2 for FP64. Five forwards and two
    backward passes are charged; each matrix has zero, one or two adjoints
    according to its source-owned constant operands. Native transform
    overhead applies to five forwards only. Every non-matrix primitive has
    an eight-primal upper for its local reverse, including accumulation/copies.
    """
    n,r=1024,384; rows=[]
    cells=diagnostics['selection_cells']; shared=diagnostics['shared_cells']
    require(set(shared)=={str(l) for l in range(16)},'Fixed sixteen shared cells')
    for l in range(16):
        cell=shared[str(l)]; total=cell['columns']; old=cell['old_columns']
        a=n if l==0 else cells[f'{l-1}:0']['count']
        c,q=cells[f'{l}:0']['count'],cells[f'{l}:1']['count']; cold=n-a
        require(0<a<=n and 0<c<=n and 0<q<=n and q<=c,'Actual current35 selector dimensions')
        require(len(cell['stops'])==l and total==(cell['stops'][-1] if l else 0),
                'Complete signed-column history')
        products=[]; native=[]
        def mm(m,k,p,rate=1,adjoints=2): products.append((m,k,p,rate,adjoints))
        # Exact compact covariance and star shapes; never all-bank dense cost
        # substituted for the actual factored old reader.
        if l==0:
            for unused in range(3): mm(512,n,512,adjoints=0)
        else:
            mm(a,a,n,adjoints=1); mm(n,a,n,adjoints=1)
            if cold:
                for unused in range(3): mm(512,cold,512,adjoints=1)
            mm(n,a,a,adjoints=1); mm(c,a,c,adjoints=1); mm(c,a,c,adjoints=1)
            if cold: mm(c,cold,c,adjoints=1)
        # Two scalar physical pushes plus the feature rho@alpha, all FP64.
        mm(n,n,1,2,1); mm(n,n,1,2,1); mm(n,n,1,2,2)
        if l<5:
            mm(n,a,total,adjoints=1)
            if total: mm(q,total,c)
        else:
            if l<15:
                before=0 if l==5 else cell['stops'][l-6]
                width=old-before
                native=[(width,a,r,1),(a,width,r,2)]
                if before: native += [(r,a,r,1),(r,r,r,2),(a,r,r,2)]
                for m,k,p,adjoints in native: mm(m,k,p,adjoints=adjoints)
                mm(r,a,width); mm(r,width,r)
                if before:
                    mm(r,a,r); mm(r,r,before); mm(r,r,r); mm(r,r,r)
            mm(n,a,r,adjoints=1); mm(n,r,old); mm(n,a,total-old,adjoints=1)
            mm(q,old,r); mm(q,r,c); mm(q,total-old,c)
        if total and l<15:
            mm(total,n,1,2); mm(total,n,1,2)  # exact omitted-source reductions
        if l:
            mm(n,64,48)
            for unused in range(6): mm(n,n,16,adjoints=1)
            mm(n,126,128); mm(n,128,128); mm(n,128,64); mm(n,64,1)
            if l<15: mm(n,64,4,adjoints=1)
        matrix=sum(2*m*k*p*rate for m,k,p,rate,adjoints in products)
        reverse=sum(2*m*k*p*rate*adjoints for m,k,p,rate,adjoints in products)
        native_extra=sum(96*(m*k+k*p+m*p) for m,k,p,adjoints in native)
        qr_forward=(4*a*r*r-4*r*r*r//3) if 5<=l<15 else 0
        qr_reverse=32*a*r*r if 5<=l<15 else 0
        # Weighted element tariff, including FP64 rate2, casts/copies and
        # checks: pair tables <=1024n²; remaining physical/BD/features <=2048n²;
        # head/selection/QR/bridge diagnostics <=1024n². Full p/pp/F traversals
        # <=64nR separately. Reverse allowance is eight per primal operation.
        pointwise=4096*n*n+64*n*total
        upper=5*matrix+2*reverse+5*native_extra+5*qr_forward+2*qr_reverse+21*pointwise
        rows.append(dict(layer=l,previous=a,live=c,response=q,columns=total,old=old,
            products=[list(v) for v in products],matrix_forward=matrix,matrix_reverse=reverse,
            native_forward_extra=native_extra,QR_forward=qr_forward,QR_reverse=qr_reverse,
            pointwise_forward=pointwise,three_call_upper=upper))
    # One original1.169GB workspace plus views, three weight/fixed-feature
    # constructors, original parameter/payload/CDF caches, full input checks,
    # cotangent/served/gradient bytes, hashes, norms and final cache retirement.
    setup_and_exports=8_000_000_000
    total=sum(row['three_call_upper'] for row in rows)+setup_and_exports
    require(total<WORK_LIMIT,'Complete three-call source envelope below8TF before numerical imports')
    return dict(layers=rows,setup_input_export_cache_upper=setup_and_exports,
        complete_source_upper=total,limit=WORK_LIMIT,actual_flopscope_bill_claimed=False,
        original_forwards=3,maximum_checkpoint_recomputed_forwards=2,backwards=2,
        parameter_updates=0,uses_actual_fixed_shape_census=True)


def prepare(source_sha256,protocol_sha256,tick=None):
    """Stdlib admission; returns inert model API and actual metadata inventories."""
    require(not {'numpy','torch','flopscope','scipy','threadpoolctl'}&set(sys.modules),
            'Fresh source admission before numerical imports')
    sources={str(Path(__file__).resolve()):source_sha256,
             str(HERE/'QUALIFICATION_PROTOCOL.md'):protocol_sha256,
             str(CAUSAL/'rangefinder_alignment.py'):ALIGNMENT_SHA}
    require(all(sha(p,tick)==h for p,h in sources.items()),'Reviewed new/frozen source owners')
    inputs={}
    old=_seals(CAUSAL/'cpu_native_y_forward_v1',NEGATIVE,False,inputs,tick)
    ow=_seals(CAUSAL/'cpu_native_y_forward_watch_v1',NEGATIVE_WATCH,False,inputs,tick)
    _join(ow,NEGATIVE)
    require(old['all_owners_retired'] and old['parameters_unchanged'] and old['full_model_calls']==1
            and old['fidelity_passed'] is False and old['fidelity']['max_abs']>1e-5
            and not old['truth_accessed'],'Old native-fidelity failure is not relabeled qualified')
    path=CAUSAL/'cpu_native_y_forward_v1/diagnostics.json'
    require(sha(path,tick)==DIAGNOSTIC_SHA==old['artifacts'][str(path)],'Actual sealed scalar census owner')
    inputs[str(path)]=DIAGNOSTIC_SHA; control=_read(path)
    vr=_seals(CAUSAL/'rangefinder_vjp_v1',VJP,True,inputs,tick)
    vw=_seals(CAUSAL/'rangefinder_vjp_watch_v1',VJP_WATCH,True,inputs,tick)
    _join(vw,VJP)
    require(vr['all_owners_retired'] and vr['rangefinder_real_algebra_VJP_qualified']
            and vr['unit_cases']==3,'Positive local product/analytic/checkpoint prerequisite only')
    fixture={WEIGHTS:'14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
        WEIGHTS.parent/'row_manifest.json':'5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
        STORE/'manifest.json':'9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'}
    for path,digest in fixture.items():
        require(sha(path,tick)==digest,'Exact canonical current35 input')
        inputs[str(path)]=digest
    spec=importlib.util.spec_from_file_location('anchored_fixed_native_Y',CAUSAL/'rangefinder_alignment.py')
    alignment=importlib.util.module_from_spec(spec); spec.loader.exec_module(alignment)
    model=alignment.model(ALIGNMENT_SHA,tick)
    sources.update(model.authenticate(OPERATOR_SHA))
    require(control['shared_source_proof']['shared_rank']==384
            and control['shared_source_proof']['full_feedback'], 'Unchanged full-causal shared384 census')
    cost=work_certificate(control)
    require(not {'numpy','torch','flopscope','scipy','threadpoolctl'}&set(sys.modules),'Complete admission remains inert')
    return dict(model=model,source_bindings=sources,input_bindings=inputs,shape_control=control,
        cost_certificate=cost,weights_path=str(WEIGHTS),candidate_sha256=C2_SHA,
        schema=SCHEMA,original_native_fidelity_passed=False,NADJ_requalified=False,
        qualification_scope='same_surrogate_execution_integrity_only',fit_authorized=False)


def _remember(torch,refs,value):
    if isinstance(value,torch.Tensor): refs.append(weakref.ref(value))
    elif isinstance(value,dict):
        for item in value.values(): _remember(torch,refs,item)
    elif isinstance(value,(tuple,list)):
        for item in value: _remember(torch,refs,item)


def _bytes(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _clear_caches(model,torch,refs):
    """Only the known admitted numerical-constant owners, never source objects."""
    co=model._JOINT._CO
    if co._PAYLOAD is not None:
        refs.extend(weakref.ref(value) for value in co._PAYLOAD.values())
        co._PAYLOAD=None
    module=sys.modules.get('gaussian_cdf_quadrature_v1')
    require(module is not None and isinstance(module,types.ModuleType),'Actual admitted CDF module')
    _remember(torch,refs,module._CONSTANTS)
    module._CONSTANTS.clear()


def evaluate(prepared,np,torch,weights,workspace,attestation,on_call=None,tick=None):
    """Numerical caller-owned scope; returns (plain result, detached bytes).

    on_call(call,record,artifacts) is after CARD/GRAPH retirement, NOT a claim
    that the shared unchanged P/cotangent/workspace has yet retired. The root
    worker publishes the sole whole-unit retired marker after this function
    returns and its own weights/workspace/backend owners have also retired.
    """
    started=time.monotonic(); model=prepared['model']; control=prepared['shape_control']
    require(prepared['schema']==SCHEMA and prepared['cost_certificate']==work_certificate(control),
            'Actual admitted fixed source envelope')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in LIMITS)
            and torch.get_num_threads()==torch.get_num_interop_threads()==1,
            'Actual nice10/sixBLAS1/TorchCPU1 execution')
    require(not {'flopscope','scipy'}&set(sys.modules) and torch.is_grad_enabled(),
            'Declared CPU proposal backend, gradients enabled outside baseline scope')
    require(type(weights) is np.ndarray and weights.shape==(16,1024,1024)
            and weights.dtype==np.float32 and np.isfinite(weights).all(),'Owned canonical typed finite weights')
    P,A=model.frozen('cpu',trainable_P=True); keys=tuple(model._JOINT.PKEYS)
    params=tuple(P[k] for k in keys); initial={k:_hash(v) for k,v in P.items()}; initial_A=_hash(A)
    require(all(v.requires_grad==(k in keys) and v.is_leaf for k,v in P.items())
            and len(keys)==11 and sum(P[k].numel() for k in keys)==45200
            and sum(P[k][0].numel() for k in model._JOINT.INACTIVE_KEYS)==66
            and not A.requires_grad and A.grad is None,
            'Exactly eleven live P leaves/45200 entries/inactive66, original A frozen')
    all_refs=[weakref.ref(v) for v in P.values()]+[weakref.ref(A)]
    # Fixed unit-L2, balanced within each layer; no RNG, labels or direction
    # selected from a measured secant. All sixteen served rows enter the loss.
    layer=torch.arange(16,dtype=torch.int64)[:,None]
    column=torch.arange(1024,dtype=torch.int64)[None,:]
    cot=(1-2*((layer+column//16)%2)).to(torch.float64)/128
    _remember(torch,all_refs,(layer,column,cot)); del layer,column
    records=[]; artifacts={}; expected_output=None; expected_gradients=None; expected_diagnostics=None

    def run_one(call):
        training=call!=0; refs=[]; began=time.monotonic()
        card=model.card(35,P=P,A=A,device='cpu',weights=weights,native_workspace=workspace,
                        native_attestation=attestation,numpy_api=np)
        refs.extend((weakref.ref(card),weakref.ref(card.coefficients),weakref.ref(card._native_rangefinder)))
        _remember(torch,refs,(card.w,card.ws,card.fixed))
        require(not card.w.requires_grad and not card.ws.requires_grad
                and not card.fixed.requires_grad,'Source-priced fixed physical weights/features')
        cstar=_hash(card.coefficients)
        source_step,source_select,source_shared=card.step,card._select,card._record_shared
        def step(*args):
            _remember(torch,refs,args[1:3])  # state/control, not shared P argument
            result=source_step(*args); _remember(torch,refs,result); return result
        def select(score):
            cell=f'{card._layer}:{card._selection_slot}'
            indices,mask=source_select(score)
            require(len(indices)==control['selection_cells'][cell]['count'],
                    'Fixed current35 shape before selected downstream arithmetic')
            _remember(torch,refs,(score,indices,mask)); return indices,mask
        def shared(layer,pp,U,F,core,old,stops):
            source_shared(layer,pp,U,F,core,old,stops)
            require(card.shared_cells[layer]==control['shared_cells'][str(layer)],'Exact fixed full-bank geometry')
            _remember(torch,refs,(pp,U,F,core))
        card.step,card._select,card._record_shared=step,select,shared
        if training:
            raw,trace=card.run(training=True)
            served=model._JOINT._CO.served(raw)
        else:
            with torch.no_grad():
                raw,trace=card.run(training=False)
                served=model._JOINT._CO.served(raw)
        _remember(torch,refs,(raw,served,trace))
        require(served.shape==(16,1024) and served.dtype==torch.float32
                and bool(torch.isfinite(served).all()) and bool((served>=0).all()),'All16 served output domains')
        output=_bytes(served); gradient_blob=b''; gradients_meta={}; scalar=None
        if training:
            loss=(served.to(torch.float64)*cot).sum(); _remember(torch,refs,loss)
            require(bool(torch.isfinite(loss)),'Finite fixed synthetic scalar')
            scalar=loss.item()
            gradients=torch.autograd.grad(loss,params,allow_unused=False,create_graph=False,retain_graph=False)
            _remember(torch,refs,gradients)
            chunks=[]; offset=0
            for key,grad in zip(keys,gradients):
                require(grad.shape==P[key].shape and grad.dtype==torch.float32
                        and not grad.requires_grad and bool(torch.isfinite(grad).all()),'Connected finite full P gradient')
                if key in model._JOINT.INACTIVE_KEYS:
                    require(bool((grad[0]==0).all()),'All66 inactive layer0 gradients exactly zero')
                blob=_bytes(grad); wide=grad.to(torch.float64)
                norm=float(torch.sqrt((wide*wide).sum()))
                require(math.isfinite(norm),'Finite full-block gradient norm')
                gradients_meta[key]=dict(shape=list(grad.shape),dtype='float32',offset=offset,nbytes=len(blob),
                    sha256=hashlib.sha256(blob).hexdigest(),norm=norm,connected=True)
                chunks.append(blob); offset+=len(blob)
            gradient_blob=b''.join(chunks)
            require(len(gradient_blob)==45200*4,'Every complete FP32 P-gradient byte retained')
            del gradients,grad,wide,loss
        diagnostic=model._JOINT._CO.check_replay(card)
        require(set(diagnostic['shared_cells'])=={str(l) for l in range(16)}
                and set(diagnostic['node_head_calls'])=={str(l) for l in range(1,15)},'All16 shared/all14 NODE cells')
        for name in ('selection_visits','variance_visits','shared_visits','node_head_calls'):
            require(all(1<=v<=(2 if training else 1) for v in diagnostic[name].values()),
                    'No unpriced third visit or missing original call: '+name)
        require(diagnostic['shared_source_proof']==control['shared_source_proof'],
                'Exactly the frozen failed-fidelity full model, not a numerical repair')
        require(all(_hash(v)==initial[k] and v.grad is None for k,v in P.items())
                and _hash(A)==initial_A and _hash(card.coefficients)==cstar,'All P/A/cstar bytes and gradients untouched')
        del card.step,card._select,card._record_shared
        record=dict(call=call,arm=ROSTER[call],rank=35,training=training,backward=training,
            scalar=scalar,served_sha256=hashlib.sha256(output).hexdigest(),gradients=gradients_meta,
            diagnostics=diagnostic,model_elapsed_s=time.monotonic()-began,
            parameters_unchanged=True,inactive66_gradient_zero=training,
            shared_parameters_retained_until_unit_end=True)
        return record,output,gradient_blob,refs

    for call in range(3):
        record,output,gradient_blob,refs=run_one(call)
        gc.collect(); require(all(ref() is None for ref in refs),'All actual call graph/card/array owners retired')
        record['call_graph_owners_retired']=True
        stable={key:record['diagnostics'][key] for key in ('selection_cells','variance_domain','shared_cells')}
        if call==0:
            expected_output=output; expected_diagnostics=stable
        else:
            require(output==expected_output and stable==expected_diagnostics,
                    'Checkpoint and fresh-repeat all16 outputs/all47 selectors/domains byte-exact to baseline')
            if call==1: expected_gradients=gradient_blob
            else: require(gradient_blob==expected_gradients,'Every full P gradient byte exactly repeats')
        produced={f'call_{call:02d}_served.f32':output}
        if call: produced[f'call_{call:02d}_gradients.f32']=gradient_blob
        artifacts.update(produced); records.append(record)
        if on_call is not None: on_call(call,record,produced)
        if tick is not None: tick()
    del params,P,A,cot
    _clear_caches(model,torch,all_refs)
    gc.collect(); require(all(ref() is None for ref in all_refs),'All helper P/A/cotangent/constant-cache arrays retired')
    require(time.monotonic()-started<120,'Whole qualification helper within120s; owner also includes admission/publication')
    return dict(schema=SCHEMA,passed=True,completed=True,rank=35,roster=list(ROSTER),
        full_model_calls=3,backwards=2,parameter_updates=0,optimizer_steps=0,records=records,
        same_surrogate_output_byte_exact=True,repeat_full_P_gradients_byte_exact=True,
        all_helper_numerical_owners_retired=True,caller_weights_workspace_retirement_still_required=True,
        sources_unchanged_by_helper=True,cost_certificate=prepared['cost_certificate'],
        elapsed_s=time.monotonic()-started,original_native_fidelity_passed=False,NADJ_requalified=False,
        native_gradient_certified=False,raw_improvement_claimed=False,truth_accessed=False,
        teacher_states_used=False,stored_H_used=False,fit_authorized=False),artifacts
