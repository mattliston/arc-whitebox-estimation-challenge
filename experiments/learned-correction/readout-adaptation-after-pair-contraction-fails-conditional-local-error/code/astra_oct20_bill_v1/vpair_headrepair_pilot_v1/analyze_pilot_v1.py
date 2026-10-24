"""Four fixed TRAIN-only actual-state local-head capacity diagnostic.

All eight original guards and byte controls are checked BEFORE target arrays.
This emits no candidate, no changed head source, and no native launch.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import argparse,gzip,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;D=H.parent;S=D/'truth_response95_v1';V=H.parents[1]/'astra_oct20_vm_frontier_v1'
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
PARENT='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
SOURCE='0315a05f5be71b235dcae0a8d2147372d8923a44e42bbcbe20f5d91ee0cfb52d'
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
PHOOK='8a94ea60aab869827ad97cec4aa2aea531bfdbc5c5341e53038b7f2feea4a91b'
SHOOK='5b3d3ba19afd6c392d6e38f1162bc7c674143b9c7c958d0a777e05630a518a67'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
    with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def load_module(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def checked_guard(out,part):
    gp=out/(part+'_GUARD.json');g=read(gp)
    assert g['successful']and g['whole_group_retired']and g['child_exit']==0 and not g['cleanup_errors']
    assert g['guard_sha256']==GUARD
    launcher=read(out/part/'launcher.json');assert launcher['client_exit']==0
    return sha(gp)
def checked_run(out,pin,diagnostic):
    d=read(out/'COMPLETE.json');rm=read(out/'RUN_MANIFEST.json')
    assert d['successful']and rm['candidate_sha256']==pin and rm['blas_threads']==4 and not rm['warmup_each']
    assert rm['setup']=='paced'and rm['max_active']==36
    return d,rm
def pred(folder,seed,diagnostic):
    name=('diagnostic_'if diagnostic else'')+'pred_%d.bin'%seed;p=folder/name
    assert p.stat().st_size==65536
    rows=[json.loads(x)for x in(folder/('diagnostic_rows.jsonl'if diagnostic else'rows.jsonl')).read_text().splitlines()if x.strip()]
    hits=[r for r in rows if int(r['rank'])==seed and r['phase']=='scored'];assert len(hits)==1
    assert hits[0]['prediction_sha256']==sha(p)
    a=np.fromfile(p,dtype='<f4').reshape(16,1024);assert np.isfinite(a).all()
    return a,hits[0],sha(p)
def capture(folder,seed,pin,hook,P,f):
    m=read(folder/'manifest.json');assert m['sha256']==pin and m['diagnostic_only']
    assert m['environment']['thread_provenance']['blas_threads']==4
    capdir=folder/'gnn_full_state_v1';meta=read(capdir/'METADATA_v1.json');done=read(capdir/'CAPTURE_COMPLETE_v1.json')
    assert meta['source_sha256']==pin and meta['hook_sha256']==hook and meta['target_free']and done['complete']
    assert done['source_sha256']==pin and done['target_free']and not done['native_handles_retained']
    cp=f.decode(meta['gnn_literal']);assert all(np.array_equal(P[k],cp[k])for k in P)
    result=[];proof=[]
    for l in range(1,16):
        rows=[r for r in done['files']if r['rank']==seed and r['phase']=='scored'and r['layer']==l];assert len(rows)==1
        r=rows[0];p=capdir/r['path'];assert sha(p)==r['sha256']and p.stat().st_size==r['bytes']
        with gzip.open(p,'rt')as z:x=json.load(z)
        assert x['rank']==seed and x['layer']==l and x['phase']=='scored'
        result.append(x);proof.append(dict(layer=l,sha256=sha(p)))
    return result,proof
def summary_values(values):
    x=np.asarray(values,dtype='f8');return dict(n=len(x),mean_pct=float(x.mean()),se_pct=float(x.std(ddof=1)/np.sqrt(len(x))),lower=int(np.sum(x<0)),per_network_pct=x.tolist())
def stats(seed,prediction,caps,truthA,truthB,W,P,fit):
    A=truthA-prediction;B=truthB-prediction
    r=dict(seed=seed,G=[],b=[],u0_sum=[],physical_gram=[],head_rms2=[]);states=[];ys=[];gates=[];physical=[];masks=[]
    for l,x in enumerate(caps,1):
        state=fit.arr(x,'state');sigma=fit.arr(x,'sigma');ng=fit.arr(x,'gate_feature');saved=fit.arr(x,'delta')
        assert state.shape==(1024,64)and sigma.shape==ng.shape==saved.shape==(1024,)
        assert np.max(np.abs(ng))<7.999
        gate=ng*float(P['scales'][l,5])+float(P['centers'][l,5]);assert gate.min()>-2e-6 and gate.max()<1+2e-6
        gate=np.clip(gate,0,1);scale=sigma*float(P['rms'][l])*(1+.25*np.tanh(float(P['gain'][l])))
        assert scale.min()>1e-12
        oracle=scale*(state@P['head'][l].astype('f8')+float(P['head_b'][l]))
        assert np.max(np.abs(oracle-saved))<1e-8+2e-5*np.max(np.abs(saved))
        mask=(prediction[l]>1e-12).astype('f8')if l<15 else np.ones(1024);X=state*mask[:,None]
        inherited=np.asarray(W[l],dtype='f8').T@np.stack([A[l-1],B[l-1]],axis=1)
        yA=(A[l]-gate*inherited[:,0])/scale;yB=(B[l]-gate*inherited[:,1])/scale
        r['G'].append(X.T@X);r['b'].append(X.T@(.5*(yA+yB)));r['u0_sum'].append(float(yA@yB))
        direct=scale[:,None]*X;r['physical_gram'].append(direct.T@direct/1024);r['head_rms2'].append(float(np.mean((saved*mask)**2)))
        states.append(X);ys.append((yA,yB));gates.append(gate);physical.append(scale);masks.append(mask)
    for k in('G','b','u0_sum','physical_gram','head_rms2'):r[k]=np.asarray(r[k])
    assert r['u0_sum'].sum()>0
    return r,dict(X=states,y=ys,gate=gates,physical=physical,mask=masks,A=A,B=B)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert sys.platform=='linux'
    pre=read(H/'PREREG_v1.json');assert sha(H/'PREREG_v1.json')=='9cd18078eed65acc4c85e107b305285bb6a148e872534d32985ef808db17248a'
    assert sha(S/'DESIGN_LOCAL_v1.json')==pre['design_sha256'];design=read(S/'DESIGN_LOCAL_v1.json');ids=pre['train_seeds']
    assert ids==[design['train'][i]for i in pre['train_positions']]and not set(ids)&set(design['protected_seed_ids'])
    fitpath=D/'local_innovation90_v1/fit_local90_v1.py';assert sha(fitpath)=='e0c431f98a071e240706f07560955ad64ada870c436d0b4da4d19efab6ad9f49'
    fit=load_module(fitpath,'vp75_root_localfit');selftests=fit.selftest()
    fp=V/'fit_frontier_v1.py';assert sha(fp)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363';f=load_module(fp,'vp75_decoder')
    src=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(src)==PARENT
    literal,_=f.literals_and_functions(src.read_text());P=f.decode(literal['_GNN_LITERAL'])
    vp=D/'old_vpair_prune_v1/candidate_msg8_old_vpair75_sci_v1.py';assert sha(vp)==SOURCE
    sl,_=f.literals_and_functions(vp.read_text());SP=f.decode(sl['_GNN_LITERAL']);assert all(np.array_equal(P[k],SP[k])for k in P)
    base=RUNS/'full_state_response_train90_v1';student=RUNS/'vpair_headrepair_pilot4_capture_v1';controls=RUNS/'vpair_headrepair_pilot4_control_v1'
    base_done,base_rm=checked_run(base,PARENT,True);sd,srm=checked_run(student,SOURCE,True);cd,crm=checked_run(controls,SOURCE,False)
    ready=read(H/'READY_v1.json');assert srm['plan_sha256']==ready['plan_capture_sha256']and crm['plan_sha256']==ready['plan_control_sha256']
    assert sha(S/'PLAN_CAPTURE_v1.json')=='a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375'
    bp=read(S/'PLAN_CAPTURE_v1.json');proof=[];inputs=[]
    # No labels above or inside this custody/parity loop.
    for i,(position,seed)in enumerate(zip(pre['train_positions'],ids),1):
        part='part%02d'%i;bpart='part%02d'%(position//3+1);assert position%3==0 and bp['groups'][position//3][0]==seed
        g=[checked_guard(out,p)for out,p in((base,bpart),(student,part),(controls,part))]
        pp,prow,ph=pred(base/bpart,seed,True);sp,srow,sh=pred(student/part,seed,True);cp,crow,ch=pred(controls/part,seed,False)
        assert sp.tobytes()==cp.tobytes(),('observer_full_prediction_parity',seed)
        assert sp[:10].tobytes()==pp[:10].tobytes(),('early_layer_parity',seed)
        bc,bproof=capture(base/bpart,seed,PARENT,PHOOK,P,f);sc,sproof=capture(student/part,seed,SOURCE,SHOOK,P,f)
        for l in range(9):
            assert np.array_equal(fit.arr(bc[l],'state'),fit.arr(sc[l],'state'))and np.array_equal(fit.arr(bc[l],'sigma'),fit.arr(sc[l],'sigma'))
        proof.append(dict(seed=seed,guards=g,parent_prediction_sha256=ph,student_prediction_sha256=sh,control_prediction_sha256=ch,
            early_prediction_bytes=True,observer_full_prediction_bytes=True,base_capture=bproof,student_capture=sproof,
            control_flops=crow['flops']))
        inputs.append((seed,pp.astype('f8'),sp.astype('f8'),bc,sc))
    parentfit=RUNS/'local_innovation90_fit_v1/FITTED_v1.json';assert sha(parentfit)=='d0ad9967ea0927ea85bfbd218da18eb020ec7768b5c634369ae38e6cb6a1f412'
    parent90=read(parentfit);baseline89={r['seed']:r for r in parent90['results']['per_layer960']['loo_per_network']}
    parent_records=[];student_records=[];details=[];target_proof=[]
    for seed,pp,sp,bc,sc in inputs:
        tr=design['rows'][str(seed)];tp=ROOT/tr['truth']['path'];assert sha(tp)==tr['truth']['sha256']and tp.stat().st_size==tr['truth']['bytes']
        wr=bp['rows'][str(seed)]['weights'];wp=ROOT/wr['path'];assert sha(wp)==wr['sha256']and wp.stat().st_size==wr['bytes']
        assert wr['raw_float32_sha256']==tr['raw_float32_sha256']and wr['raw_float32_sha256']not in design['protected_raw_weight_hashes']
        with np.load(tp,allow_pickle=False)as z:
            assert int(z['seed'])==seed and str(z['weights_sha256'])==wr['raw_float32_sha256'];A=z['mc_post_A'].astype('f8');B=z['mc_post_B'].astype('f8')
        W=np.load(wp,allow_pickle=False,mmap_mode='r');assert W.shape==(16,1024,1024)
        br,bd=stats(seed,pp,bc,A,B,W,P,fit);sr,st=stats(seed,sp,sc,A,B,W,P,fit)
        # Prove the reused89-network LOO result has exactly this local-risk denominator.
        baseline_u0=float(br['u0_sum'].sum()/(15*1024))
        np.testing.assert_allclose(baseline_u0,baseline89[seed]['u0'],rtol=1e-11,atol=1e-15)
        parent_records.append(br);student_records.append(sr);layers=[]
        for l in range(1,16):
            drift=sp[l]-pp[l];local=drift-st['gate'][l-1]*(np.asarray(W[l],dtype='f8').T@(sp[l-1]-pp[l-1]))
            energy=float(np.mean(drift*drift));locenergy=float(np.mean(local*local))
            layers.append(dict(layer=l,prediction_drift_rms=energy**.5,prediction_drift_mean=float(drift.mean()),
                prediction_drift_constant_fraction=float(drift.mean()**2/energy)if energy else 0.,
                local_drift_rms=locenergy**.5,local_drift_mean=float(local.mean()),
                local_drift_constant_fraction=float(local.mean()**2/locenergy)if locenergy else 0.))
        terminal_base=float(np.mean((A[-1]-pp[-1])*(B[-1]-pp[-1])));terminal_student=float(np.mean((A[-1]-sp[-1])*(B[-1]-sp[-1])))
        assert terminal_base>0 and terminal_student>0
        details.append(dict(seed=seed,layers=layers,unadapted_terminal_train_relative_pct=100*(terminal_student/terminal_base-1),student_details=st))
        target_proof.append(dict(seed=seed,truth_sha256=sha(tp),weights_sha256=sha(wp)))
    def crossvalidate(records):
        G=sum(r['G']for r in records);b=sum(r['b']for r in records);cv=[];coefs=[]
        for r in records:
            c=fit.frozen(fit.coefficients(G-r['G'],b-r['b'],3,'per_layer960'),P);cv.append(fit.metric(r,c));coefs.append(c)
        full=fit.frozen(fit.coefficients(G,b,4,'per_layer960'),P)
        corr=np.stack([np.einsum('li,lij,lj->l',full,r['physical_gram'],full)for r in records]);orig=np.stack([r['head_rms2']for r in records])
        ratio=np.sqrt(np.maximum(corr.sum(axis=0),0)/np.maximum(orig.sum(axis=0),1e-30))
        return cv,coefs,dict(overall=float(np.sqrt(max(0,corr.sum())/max(orig.sum(),1e-30))),max_layer=float(ratio.max()),per_layer=ratio.tolist())
    bcv,_,_=crossvalidate(parent_records);scv,scoef,physical=crossvalidate(student_records)
    factor=pre['cost_factor'];vs4=summary_values([100*(factor*s['u1']/b['u1']-1)for s,b in zip(scv,bcv)])
    vs89=summary_values([100*(factor*s['u1']/baseline89[s['seed']]['u1']-1)for s in scv])
    for detail,c in zip(details,scoef):
        st=detail.pop('student_details');detail['heldout_local_correction_rms_by_layer']=[float(np.sqrt(np.mean((scale*(X@cc))**2)))for scale,X,cc in zip(st['physical'],st['X'],c)]
    cv=fit.summary(scv);g=pre['gates']
    checks=dict(v75_LOO_improves=cv['mean_relative_pct']<=g['v75_LOO_mean_local_improvement_pct_lte']and cv['lower']>=g['v75_LOO_lower_networks_gte'],
        versus_equal_training_count=vs4['mean_pct']<=g['same_training_count_billweighted_local_proxy_vs_parent4_lte']and vs4['lower']>=g['same_training_count_billweighted_local_proxy_lower_networks_gte'],
        versus_parent89_LOO=vs89['mean_pct']<=g['conservative_billweighted_local_proxy_vs_parent89_lte'],
        physical_step=physical['overall']<=g['overall_physical_correction_to_head_rms_lte']and physical['max_layer']<=g['max_layer_physical_correction_to_head_rms_lte'])
    result=dict(utc=utc(),status='CAPACITY_PILOT_PASS_LEAD_REVIEW_REQUIRED'if all(checks.values())else'CLOSE_PILOT_GATE_FAIL',
        checks=checks,student_LOO=cv,student_LOO_rows=scv,parent4_LOO=fit.summary(bcv),parent4_LOO_rows=bcv,
        billweighted_LOCAL_proxy_vs_parent4=vs4,billweighted_LOCAL_proxy_vs_parent89=vs89,
        comparison_warning='Local normalized innovation risks with a prior whole-network bill factor; not actual adjusted scores. Parent89 uses89 other TRAIN networks; student uses3.',
        physical_step=physical,drift=details,parity_provenance=proof,targets=target_proof,
        source_sha256=SOURCE,reference_source_sha256=PARENT,prereg_sha256=sha(H/'PREREG_v1.json'),script_sha256=sha(__file__),
        parent_fit_sha256=sha(parentfit),selftests=selftests,no_candidate_exported=True,no_protected_truth_read=True,
        native_feedback_not_modeled=True)
    a.out.mkdir(exist_ok=False);write(a.out/'RESULT_v1.json',result)
    print(json.dumps({k:result[k]for k in('utc','status','checks','student_LOO','billweighted_LOCAL_proxy_vs_parent4','billweighted_LOCAL_proxy_vs_parent89','physical_step')}),flush=True)
if __name__=='__main__':main()
