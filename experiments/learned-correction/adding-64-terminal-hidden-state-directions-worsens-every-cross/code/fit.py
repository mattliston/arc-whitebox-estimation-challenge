"""Fixed target-free enc4 complement, five network folds, one ridge and export."""
from pathlib import Path
import ast,base64,hashlib,importlib.util,json,os,sys,time,zlib

D=Path(__file__).resolve().parent
RIDGE=.001
GATES=dict(raw_pct_max=-.2,adjusted_pct_max=-.2,fold_adjusted_wins_min=4)
ADDED_BILL=300000
PARENT_SHA='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')

def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def checked(p):
    assert pin(p['path'])==p
    return Path(p['path'])

def once(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def read(p):return json.loads(Path(p).read_text())

def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def job(c,pp,mode):
    cmd=[c['runtime'],'-B',str(Path(__file__).resolve()),'--'+mode,pp['path'],pp['sha256']]
    return dict(command=cmd,guard_command=[c['light'],'-B',c['guard_source']['path'],'--record',str(D/(mode+'_guard.json')),
        '--seconds','120','--rss-limit','7500000000',*cmd],log=str(D/(mode+'.log')))

def successful_guard(c,pp,mode):
    p=pin(D/(mode+'_guard.json'));g=read(checked(p))
    assert g['command']==job(c,pp,mode)['command'] and g['completed'] and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
    assert g['rss_trigger_bytes']==7500000000 and 0<g['max_rss_bytes']<7500000000
    assert 0<g['elapsed_s']<=120 and g['samples']>0
    return p

def prepare(pp,c):
    seal=dict(protocol=pp,fit_source=pin(__file__),capture_source=pin(D/'prepare_capture.py'),ridge=RIDGE,
        fold_count=5,cv_gates=GATES,added_bill_flops=ADDED_BILL,target='truth_minus_current_terminal_prediction',
        source_frozen_before_targets=True,coefficient_count=64,intercept=False)
    once(D/'fit_seal.json',seal)
    once(D/'fit_commands.json',dict(protocol=pp,seal=pin(D/'fit_seal.json'),jobs={k:job(c,pp,k) for k in ('cv','export')},thread_env=dict.fromkeys(THREADS,'1')))
    print(json.dumps(seal))

def export_source(parent,vector):
    old='    return delta,state\n'
    body=("    if l == 15 and _T128_VECTOR is not None:\n"
          "        extra = (((sigma.astype(fnp.float32) * c['rms']) * (h @ _T128_VECTOR)) * c['gain']).astype(fnp.float32)\n"
          "        delta = (delta + extra).astype(fnp.float32)\n" + old)
    tree=ast.parse(parent);n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gnn_step')
    lines=parent.splitlines(keepends=True);before=''.join(lines[n.lineno-1:n.end_lineno]);assert before.count(old)==1
    after=before.replace(old,body,1);assert parent.count(before)==1
    suffix=("\n\n# Fixed terminal128 complement; no change to recurrent state or old head.\n"
            "_T128_LITERAL = " + repr(vector) + "\n_T128_VECTOR = None\n_T128_PARENT_SETUP = Estimator.setup\n"
            "def _t128_setup(self, context):\n    global _T128_VECTOR\n"
            "    result = _T128_PARENT_SETUP(self, context)\n"
            "    _T128_VECTOR = None if _T128_LITERAL is None else fnp.asarray(_T128_LITERAL, dtype=fnp.float32)\n"
            "    return result\nEstimator.setup = _t128_setup\n")
    source=parent.replace(before,after,1)+suffix
    assert source[:-len(suffix)].replace(after,before,1)==parent
    compile(source,'terminal128_candidate','exec')
    return source,dict(edits=[dict(old=before,new=after)],appended_suffix=suffix,whole_source_inverse_exact=True,
        existing_GNN_and_physics_unchanged=True,terminal_layer_only=15,disabled_literal_none_preserves_original_arithmetic=True)

def perform(mode,pp,c,m):
    m.environment(c);assert all(os.environ.get(k)=='1' for k in THREADS)
    assert str(Path(sys.executable).absolute())==c['runtime']
    seal_pin=pin(D/'fit_seal.json');seal=read(checked(seal_pin));commands=read(D/'fit_commands.json')
    assert seal['protocol']==pp and seal['fit_source']==pin(__file__) and seal['capture_source']==pin(D/'prepare_capture.py')
    assert seal['ridge']==RIDGE and seal['cv_gates']==GATES and seal['added_bill_flops']==ADDED_BILL
    assert commands['protocol']==pp and commands['seal']==seal_pin
    assert commands['jobs']=={k:job(c,pp,k) for k in ('cv','export')}
    inputs=m.capture_inputs(c,pp);assert len(inputs)==20
    parent=c['parent'];assert parent['sha256']==PARENT_SHA
    parent_source=checked(parent).read_text()
    if mode=='export':
        cv_guard=successful_guard(c,pp,'cv');cv_pin=pin(D/'cv_result.json');cv=read(checked(cv_pin))
        assert cv['completed'] and cv['passed_integrity'] and cv['admitted'] and cv['all_finite'] and cv['fit_count']==5
        assert cv['protocol']==pp and cv['fit_source']==pin(__file__) and cv['fit_seal']==seal_pin
        assert cv['cv_gates']==GATES and cv['ridge']==RIDGE and cv['inputs']==inputs and cv['added_bill_flops']==ADDED_BILL
    import numpy as np
    from threadpoolctl import threadpool_info
    assert np.__version__=='2.4.6' and all(v['num_threads']==1 for v in threadpool_info())
    started=time.monotonic();literal=next(ast.literal_eval(n.value) for n in ast.parse(parent_source).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_GNN_LITERAL' for t in n.targets))
    def param(k):
        shape,s=literal[k];return np.frombuffer(zlib.decompress(base64.b64decode(s)),dtype=np.float32).reshape(shape).copy()
    A=param('enc4_w').astype(np.float64);assert A.shape==(64,128)
    _,singular,Vh=np.linalg.svd(A,full_matrices=True);tol=np.finfo(np.float64).eps*128*singular[0]
    rank=int(np.sum(singular>tol));assert rank==64
    Q=Vh[rank:].T.copy();assert Q.shape==(128,64)
    for j in range(64):
        if Q[int(np.argmax(np.abs(Q[:,j]))),j]<0:Q[:,j]*=-1
    assert np.linalg.norm(A@Q)<1e-10*max(1.,float(np.linalg.norm(A))) and np.linalg.norm(Q.T@Q-np.eye(64))<1e-10
    basis_sha=hashlib.sha256(Q.tobytes()).hexdigest()
    if mode=='export':assert basis_sha==cv['basis_sha256']
    basis=dict(rank=rank,rank_tolerance=float(tol),smallest_singular_value=float(singular[-1]),largest_singular_value=float(singular[0]),
        norm_AQ=float(np.linalg.norm(A@Q)),orthogonality_error=float(np.linalg.norm(Q.T@Q-np.eye(64))),target_free=True)
    head,bias=param('head')[15],param('head_b')[15];enc4=param('enc4_w');enc4_b=param('enc4_b')
    def replay(packet,vector=None):
        base=((packet['sigma'].astype(np.float32)*np.float32(packet['rms']))*(packet['hidden']@head+bias))
        base=(base*packet['effective_gain']).astype(np.float32)
        if vector is not None:
            extra=((packet['sigma'].astype(np.float32)*np.float32(packet['rms']))*(packet['h128']@vector))
            extra=(extra*packet['effective_gain']).astype(np.float32);base=(base+extra).astype(np.float32)
        return np.maximum(packet['host'].astype(np.float64)+base.astype(np.float64),0).astype(np.float32),base
    data=[]
    for i,item in enumerate(inputs):
        row=item['row'];assert row['rank']==c['ranks'][i] and row['weights']==c['data'][i]['weights']
        assert row['source']==parent and not row['targets_accessed']
        with np.load(checked(row['packet']),allow_pickle=False) as z:p={k:np.array(z[k],copy=True) for k in z.files}
        assert p['h128'].shape==(1024,128) and p['h128'].dtype==np.float32 and p['hidden'].shape==(1024,64)
        assert p['enc4_w'].tobytes()==enc4.tobytes() and p['enc4_b'].tobytes()==enc4_b.tobytes()
        reconstructed=p['h128']@enc4.T+enc4_b
        assert reconstructed.tobytes()==p['hidden'].tobytes()
        pred,delta=replay(p);assert delta.tobytes()==p['delta'].tobytes()
        assert pred.astype(np.float64).tobytes()==p['prediction'][-1].tobytes()
        physical=(p['sigma'].astype(np.float32)*np.float32(p['rms'])).astype(np.float64)*p['effective_gain'].astype(np.float64)
        B=physical[:,None]*(p['h128'].astype(np.float64)@Q)
        assert B.shape==(1024,64) and np.isfinite(B).all() and 0<row['flops']<2**41-ADDED_BILL
        data.append(dict(packet=p,B=B,baseline=pred.astype(np.float64),flops=row['flops']))
    if mode=='cv':
        once(D/'features_frozen_before_targets.json',dict(protocol=pp,fit_source=pin(__file__),inputs=inputs,
            basis=basis,basis_sha256=basis_sha,all_baseline_readouts_exact=True,all_h128_state_replays_exact=True,targets_decoded=False))
    targets=[]
    for fixture in c['data']:
        with np.load(checked(fixture['truth']),allow_pickle=False) as z:y=z['mc_post'][-1].astype(np.float64)
        assert y.shape==(1024,) and np.isfinite(y).all();targets.append(y)
    def fit(indices):
        B=np.concatenate([data[i]['B'] for i in indices]);y=np.concatenate([targets[i]-data[i]['baseline'] for i in indices])
        scale=np.sqrt(np.mean(B*B,axis=0));assert np.isfinite(scale).all() and np.min(scale)>1e-20
        X=B/scale;gram=X.T@X/len(X)+RIDGE*np.eye(64);rhs=X.T@y/len(X)
        gamma=np.linalg.solve(gram,rhs);beta=gamma/scale;v=(Q@beta).astype(np.float32)
        error=float(np.linalg.norm(gram@gamma-rhs)/max(float(np.linalg.norm(rhs)),1e-30));assert error<1e-9
        null=float(np.linalg.norm(A@v.astype(np.float64)));denom=float(np.linalg.norm(A)*np.linalg.norm(v.astype(np.float64)))
        assert np.isfinite(v).all() and null<=1e-6*max(denom,1e-30)
        return v,dict(normal_equation_relative_error=error,gram_condition=float(np.linalg.cond(gram)),
            norm_Av=null,relative_norm_Av=null/max(denom,1e-30),vector_norm=float(np.linalg.norm(v.astype(np.float64))),training_rows=len(X))
    if mode=='export':
        vector,equations=fit(list(range(20)));source,proof=export_source(parent_source,vector.tolist())
        # The disabled branch is source-reviewable; it is not an additional model trial.
        export_source(parent_source,None)
        path=D/'candidate.py'
        with path.open('x') as f:f.write(source)
        candidate=pin(path);delta=dict(parent=parent,candidate=candidate,**proof);once(D/'source_delta.json',delta)
        out=dict(completed=True,passed_integrity=True,protocol=pp,fit_source=pin(__file__),fit_seal=seal_pin,
            cv_result=cv_pin,cv_guard=cv_guard,parent=parent,candidate=candidate,source_inverse=proof,source_delta=pin(D/'source_delta.json'),
            coefficient_vector=vector.tolist(),added_bill_flops=ADDED_BILL,basis=basis,basis_sha256=basis_sha,equations=equations,
            inputs=inputs,all_existing_baseline_replays_exact=True,fit_count=1,all_finite=True,ridge=RIDGE,
            actual_native_bill_and_accuracy_unmeasured=True,bank_qualified=False,elapsed_s=time.monotonic()-started)
        once(D/'fit_export.json',out);print(json.dumps(dict(exported=candidate,equations=equations)));return
    predictions={};folds=[]
    for fold in range(5):
        train=[i for i in range(20) if i%5!=fold];held=[i for i in range(20) if i%5==fold]
        vector,equations=fit(train)
        for i in held:
            predictions[i]=replay(data[i]['packet'],vector)[0].astype(np.float64);assert np.isfinite(predictions[i]).all()
        folds.append(dict(fold=fold,train=[c['ranks'][i] for i in train],held=[c['ranks'][i] for i in held],equations=equations,coefficient_vector=vector.tolist()))
    rows=[]
    for i,rank in enumerate(c['ranks']):
        base=float(np.mean((data[i]['baseline']-targets[i])**2));new=float(np.mean((predictions[i]-targets[i])**2))
        assert 0<base and np.isfinite(new)
        rows.append(dict(rank=rank,fold=i%5,base_raw=base,new_raw=new,base_adjusted=base*max(.1,data[i]['flops']/2**41),
            new_adjusted=new*max(.1,(data[i]['flops']+ADDED_BILL)/2**41),parent_flops=data[i]['flops'],candidate_proxy_flops=data[i]['flops']+ADDED_BILL))
    def pct(selected,key):return 100*(sum(r['new_'+key] for r in selected)/sum(r['base_'+key] for r in selected)-1)
    fr=[pct([r for r in rows if r['fold']==f],'raw') for f in range(5)]
    fa=[pct([r for r in rows if r['fold']==f],'adjusted') for f in range(5)]
    metrics=dict(raw_pct=pct(rows,'raw'),adjusted_pct=pct(rows,'adjusted'),fold_raw_pct=fr,fold_adjusted_pct=fa,
        fold_raw_wins=sum(v<0 for v in fr),fold_adjusted_wins=sum(v<0 for v in fa))
    admitted=metrics['raw_pct']<=GATES['raw_pct_max'] and metrics['adjusted_pct']<=GATES['adjusted_pct_max'] and metrics['fold_adjusted_wins']>=GATES['fold_adjusted_wins_min']
    out=dict(completed=True,passed_integrity=True,admitted=admitted,protocol=pp,fit_source=pin(__file__),fit_seal=seal_pin,
        cv_gates=GATES,ridge=RIDGE,fit_count=5,all_finite=True,metrics=metrics,inputs=inputs,rows=rows,folds=folds,
        basis=basis,basis_sha256=basis_sha,added_bill_flops=ADDED_BILL,all_existing_baseline_replays_exact=True,
        historically_reused_TRAIN=True,adjusted_is_conservative_bill_proxy=True,actual_native_candidate_unmeasured=True,
        full_data_fit_performed=False,selection_or_tuning=False,bank_qualified=False,elapsed_s=time.monotonic()-started)
    once(D/'cv_result.json',out);print(json.dumps(dict(admitted=admitted,metrics=metrics,basis=basis)))

if __name__=='__main__':
    assert len(sys.argv)==4 and sys.argv[1] in ('--prepare','--cv','--export')
    m=module(D/'prepare_capture.py','terminal128_acquisition');pp,c=m.plan(sys.argv[2],sys.argv[3])
    if sys.argv[1]=='--prepare':prepare(pp,c)
    else:perform(sys.argv[1][2:],pp,c,m)
