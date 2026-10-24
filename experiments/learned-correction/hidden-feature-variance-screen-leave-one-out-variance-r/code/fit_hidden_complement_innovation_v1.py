"""One fixed shared hidden2-complement correction to current777a local innovations.

All learned projection/ridge coefficients use TRAIN4 only. Validation truth is
opened only after deployable values are frozen. Recurrent state is unchanged.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import fit_frontier_v1 as f
import fit_current_msg8_messages_v2 as current

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace'); RUNS=Path('/home/arcworker/arc-runs')
TRAIN=(35,150,200,250); VALID=(259,277,287)
RIDGE=.1; DAMPING=.5
PLAN=HERE/'PLAN_BANK8_ACTUAL_SHARED_v1.json'
PLAN_SHA='cf76262261a85fcb6b5c508ad37a2b13f92884b6b69aff66bd0da4bcd0833793'
PARENTS={role:HERE/'msg8_loaderfix_v1'/('candidate_msg8_teacher_loaderfix_'+suffix+'_v1.py') for role,suffix in [('science','sci'),('release','rel')]}
PINS={'science':current.SCI,'release':current.REL}
ROOT_TARGET=HERE.parent/'astra_oct20_bill_v1/shared_innovation_head_v1/fit_v1.py'
ROOT_TARGET_SHA='5e707f9d2a185eb34a5e2390c20b063f45bfc17edb4998b20891a5384a8d1f58'
sha=f.sha; utc=f.utc; read=lambda p:json.loads(Path(p).read_text())


def null_basis(P):
    w=P['enc4_w'].astype('f8'); assert w.shape==(64,128)
    _,sing,vh=np.linalg.svd(w,full_matrices=True)
    assert sing[-1]>1e-8*sing[0]
    basis=vh[64:].T
    np.testing.assert_allclose(w@basis,0,atol=1e-11,rtol=0)
    np.testing.assert_allclose(basis.T@basis,np.eye(64),atol=1e-12,rtol=0)
    return basis


def fit(records,P,basis):
    # Residualization sees training inputs only, independently for every LOO fold.
    S=np.concatenate([r['state'] for r in records])
    H=np.concatenate([r['hidden2'] for r in records])
    mask=np.concatenate([r['mask'] for r in records])
    X=np.column_stack((S,np.ones(len(S))))*mask[:,None]
    Z=(H@basis)*mask[:,None]
    B=np.linalg.lstsq(X,Z,rcond=1e-10)[0]
    R=Z-X@B
    orth=float(np.linalg.norm(X.T@R)/max(np.linalg.norm(X)*np.linalg.norm(R),1e-30))
    assert orth<1e-8
    y=np.concatenate([.5*(r['A']+r['B']) for r in records])
    scale=np.maximum(np.sqrt(np.mean(R*R,axis=0)),1e-8)
    z=R/scale
    c=DAMPING*np.linalg.solve(z.T@z+RIDGE*len(z)*np.eye(64),z.T@y)/scale
    qh=(basis@c).astype('f4'); qs=(-B[:64]@c).astype('f4'); qb=np.float32(-B[64]@c)
    Q={k:v.copy() for k,v in P.items()}
    Q['head'][1:16]=(P['head'][1:16].astype('f8')+qs).astype('f4')
    Q['head_b'][1:16]=(P['head_b'][1:16].astype('f8')+qb).astype('f4')
    out=dict(hidden=qh,parameters=Q,projection=B,coefficient=c,feature_scale=scale,
             training_projection_orthogonality=orth,training_rows=len(y),training_ranks=sorted({r['rank'] for r in records}))
    got=np.concatenate([correction(r,out,P) for r in records])
    expected=R@c
    err=float(np.max(np.abs(got-expected)))
    assert err<2e-6+2e-5*float(np.max(np.abs(expected)))
    out['float32_export_max_abs_error']=err
    return out


def correction(r,model,P):
    l=r['layer'];Q=model['parameters']
    return r['mask']*(r['hidden2']@model['hidden'].astype('f8')+
        r['state']@(Q['head'][l].astype('f8')-P['head'][l].astype('f8'))+
        float(Q['head_b'][l])-float(P['head_b'][l]))


def metric(records,model,P):
    nets=[]
    for rank in sorted({r['rank'] for r in records}):
        rr=[r for r in records if r['rank']==rank]
        a=np.concatenate([r['A'] for r in rr]);b=np.concatenate([r['B'] for r in rr])
        pred=np.concatenate([correction(r,model,P) for r in rr])
        u0=float(np.mean(a*b));u1=float(np.mean((a-pred)*(b-pred)))
        nets.append(dict(rank=rank,u0=u0,u1=u1,relative_pct=100*(u1/u0-1) if u0>0 else None))
    good=all(x['u0']>0 for x in nets)
    return dict(networks=nets,positive_baselines=good,lower=sum(x['u1']<x['u0'] for x in nets),
                mean_pct=float(np.mean([x['relative_pct'] for x in nets])) if good else None)


def physical(records,model,P):
    def one(rr):
        pred=np.concatenate([correction(r,model,P)*r['physical'] for r in rr])
        old=np.concatenate([r['saved_delta'] for r in rr])
        output=np.concatenate([r['mean_output'] for r in rr])
        rms=lambda x:float(np.sqrt(np.mean(x*x)))
        return dict(n=len(pred),correction_rms=rms(pred),correction_max_abs=float(np.max(np.abs(pred))),
                    correction_mean=float(np.mean(pred)),old_head_rms=rms(old),output_rms=rms(output),
                    correction_rms_over_old_head_pct=100*rms(pred)/max(rms(old),1e-30),
                    correction_rms_over_output_pct=100*rms(pred)/max(rms(output),1e-30))
    return dict(total=one(records),by_layer=[dict(layer=l,**one([r for r in records if r['layer']==l])) for l in range(1,16)])


def load_records(selected,rows,P,plan):
    # Same pinned local-innovation definition and row/gate/head checks as root5e707f9d.
    bank=RUNS/'msg8_loaderfix337422_control32_markline_blas4_v1'
    assert read(bank/'COMPLETE.json')['successful']
    rankgroups={rank:i+1 for i,g in enumerate(plan['groups']) for rank in g}
    records=[];provenance=[];checks=[]
    for rank in selected:
        group='part%02d'%rankgroups[rank];folder=bank/group
        guard=read(bank/(group+'_GUARD.json'));manifest=read(folder/'manifest.json')
        assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0 and not guard['cleanup_errors']
        assert guard['guard_sha256']=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
        assert manifest['sha256']==current.SCI and manifest['environment']['thread_provenance']['blas_threads']==4
        entry=next(x for x in map(json.loads,(folder/'rows.jsonl').read_text().splitlines()) if x['rank']==rank)
        pp=folder/('pred_%d.bin'%rank);assert pp.stat().st_size==65536 and sha(pp)==entry['prediction_sha256']
        pred=np.fromfile(pp,dtype='<f4').reshape(16,1024).astype('f8');assert np.isfinite(pred).all()
        data={};files=[]
        for kind in ('weights','truth_A','truth_B'):
            item=plan['rows'][str(rank)][kind];path=ROOT/item['path']
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
            if kind=='weights':
                data[kind]=np.load(path,allow_pickle=False,mmap_mode='r');assert data[kind].shape==(16,1024,1024)
            else:
                with np.load(path,allow_pickle=False) as z:data[kind]=z[item['key']].astype('f8')
                assert data[kind].shape==(16,1024) and np.isfinite(data[kind]).all()
            files.append(dict(kind=kind,path=str(path),sha256=sha(path)))
        for r in [x for x in rows if x['rank']==rank]:
            l=r['layer'];ids=np.asarray(r['rows']);H=f.array(r,'H').astype('f8')
            assert np.max(np.abs(H[:,5]))<7.999,'Cannot invert clipped gate feature'
            gate=H[:,5]*float(P['scales'][l,5])+float(P['centers'][l,5])
            assert np.min(gate)>-2e-6 and np.max(gate)<1+2e-6
            gate=np.clip(gate,0,1)
            state=f.array(r,'state').astype('f8');hidden=f.array(r,'hidden2').astype('f8')
            assert state.shape==(128,64) and hidden.shape==(128,128)
            gain=1+.25*np.tanh(float(P['gain'][l]))
            physical=f.array(r,'sigma').astype('f8')*float(P['rms'][l])*gain
            assert physical.shape==(128,) and np.min(physical)>1e-12
            saved=f.array(r,'delta').astype('f8')
            rec=physical*(state@P['head'][l].astype('f8')+float(P['head_b'][l]))
            err=float(np.max(np.abs(rec-saved)));assert err<1e-8+2e-5*float(np.max(np.abs(saved)))
            mask=(pred[l,ids]>1e-12).astype('f8') if l<15 else np.ones(128)
            values={}
            W=np.asarray(data['weights'][l],dtype='f8')
            for half in ('A','B'):
                residual=data['truth_'+half]-pred
                inherited=gate*(W.T@residual[l-1])[ids]
                values[half]=(residual[l,ids]-inherited)/physical
            records.append(dict(rank=rank,layer=l,state=state,hidden2=hidden,mask=mask,physical=physical,
                saved_delta=saved,mean_output=pred[l,ids],**values))
            checks.append(dict(rank=rank,layer=l,head_replay_max_abs=err,clipped_rows=int(np.sum(mask==0))))
        provenance.append(dict(rank=rank,prediction_sha256=sha(pp),guard_sha256=sha(bank/(group+'_GUARD.json')),files=files))
    return records,provenance,checks


def export(out,model,P,functions,report_sha):
    native=out/'native_v1';native.mkdir(exist_ok=False)
    q=model['hidden'].tolist();Q=model['parameters'];name='msg8_hidden_complement_innovation'
    step=functions['_gop2_gnn_step'];old="(state @ c['head'])"
    assert step.count(old)==1
    step=step.replace(old,"((state @ c['head']) + _vmfr_hidden_complement(h))")
    helper='''
def _vmfr_hidden_complement(h):
    global _VMFR_HIDDEN_COMPLEMENT_VECTOR
    if _VMFR_HIDDEN_COMPLEMENT_VECTOR is None:
        _VMFR_HIDDEN_COMPLEMENT_VECTOR=fnp.asarray(_VMFR_HIDDEN_COMPLEMENT_LIST,dtype=fnp.float32)
    return h @ _VMFR_HIDDEN_COMPLEMENT_VECTOR
'''
    tail='\n# Shared hidden2 complement local-innovation correction, fixed TRAIN4.\n'
    tail+='assert _GNN_PARAMS is None and not _E2_GNN_STATE and not _E2_MSG\n'
    tail+='assert _VMFR_CM == 8 and _O20_FLATEXTRA_ON is False\n'
    tail+='_GNN_LITERAL = '+repr(f.encode(Q))+'\n'
    tail+='_VMFR_HIDDEN_COMPLEMENT_LIST = '+repr(q)+'\n_VMFR_HIDDEN_COMPLEMENT_VECTOR = None\n'
    tail+=helper+'\n'+step+'\n_FAST_RESCUE_FRAC = {}\n'
    tp=native/('tail_'+name+'_v1.py');f.write(tp,tail)
    oldspec=next(c for c in read(HERE/'msg8_loaderfix_v1/CANDIDATES_LINUX_v1.json')['candidates'] if c['name']=='msg8_teacher_loaderfix')
    oldtail=Path(oldspec['tail']);assert sha(oldtail)==oldspec['tail_sha256']
    fixture=native/('composed_fixture_'+name+'_v1.py');f.write(fixture,oldtail.read_text()+tail)
    c=dict(name=name,kind='fixed_shared_hidden2_complement_innovation',message_width=8,
           reference_submission=337422,unchanged_state_and_recurrent_weights=True,
           append_tail=str(tp),append_tail_sha256=sha(tp),tail=str(fixture),tail_sha256=sha(fixture),
           fixture_not_deployment=True,fit_report_sha256=report_sha)
    for role,parent in PARENTS.items():
        sp=native/('candidate_'+name+('_sci_v1.py' if role=='science' else '_rel_v1.py'))
        text=parent.read_text()+tail;compile(text,str(sp),'exec');f.write(sp,text)
        c[role]=dict(path=str(sp),sha256=sha(sp),effective_sha256=sha(sp),parent_path=str(parent),parent_sha256=sha(parent))
    f.jwrite(native/'CANDIDATES_v1.json',dict(utc=utc(),candidates=[c],reference_submission=337422))
    f.jwrite(native/'TINY_WITH_EXACT_PARENT_v1.json',dict(candidates=[oldspec,c],reference_submission=337422))
    return c


def selftest():
    rng=np.random.default_rng(20261021)
    P=dict(enc4_w=rng.normal(size=(64,128)).astype('f4'),head=np.zeros((16,64),'f4'),head_b=np.zeros(16,'f4'))
    basis=null_basis(P);hh=rng.normal(size=(1200,128));st=hh@P['enc4_w'].T;mask=np.ones(1200)
    residual=hh@basis
    cv=rng.normal(size=64)*.01;y=residual@cv
    rr=[dict(rank=35,layer=1,state=st,hidden2=hh,mask=mask,A=y,B=y)]
    m=fit(rr,P,basis);score=metric(rr,m,P)
    assert score['mean_pct']<-50 and m['training_projection_orthogonality']<1e-8
    np.testing.assert_allclose(P['enc4_w']@m['hidden'],0,atol=2e-7,rtol=0)
    return dict(training_projection_only=True,nullspace_identity=True,ridge_recovery=True,float32_fold_identity=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    assert sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    assert sha(current.__file__)=='56d3c63e1d766e6b1870e6aa07b88b84c02172d71edf28a6ad6a1724ad3f53ea'
    checks=selftest()
    if a.self_test:print(json.dumps(checks));return
    assert sys.platform=='linux' and a.out is not None and sha(PLAN)==PLAN_SHA and sha(ROOT_TARGET)==ROOT_TARGET_SHA
    for role,path in PARENTS.items():assert sha(path)==PINS[role]
    a.out.mkdir(exist_ok=False,parents=True)
    f.jwrite(a.out/'PREREG_v1.json',dict(utc=utc(),kind='SHARED_HIDDEN2_COMPLEMENT_LOCAL_INNOVATION',
        train=TRAIN,validation=VALID,layers=list(range(1,16)),training_rows=7680,ridge=RIDGE,damping=DAMPING,
        design='Fixed original enc4 nullspace64; TRAIN-only masked least-squares residualization against state64+constant; one shared64 ridge vector',
        target='Exact root5e707f9d local innovation divided by sigma*rms*gain; both pinned truth halves averaged for fitting',
        loo='Recompute target-free residualization and fixed ridge independently on each other3 training networks',
        validation_rule='Open validation3 truth only after float32 deployable readouts freeze; no tuning or selection',
        gate='Positive cross-half innovation baselines; LOO mean<0 with >=3/4 lower; validation mean<=-.1% with>=2/3 lower',
        physical_trust='Report actual TRAIN correction RMS/max and ratios; no arbitrary parameter-coordinate norm cap',
        state_and_birth_unchanged=True,no_held25_or_fresh_reads=True,no_adjusted_gain_claim=True,
        source_sha256=sha(__file__),root_target_sha256=ROOT_TARGET_SHA,source_pins=PINS,plan_sha256=sha(PLAN),selftests=checks))
    rows,paths,meta=current.collect(RUNS/'current_msg8_gnn_capture_v1')
    literals,functions=f.literals_and_functions(PARENTS['science'].read_text());P=f.decode(literals['_GNN_LITERAL'])
    captured=f.decode(meta['gnn_literal']);assert all(np.array_equal(P[k],captured[k]) for k in P)
    trainrows=[r for r in rows if r['rank'] in TRAIN];replay=current.matrices(trainrows,P)
    basis=null_basis(P);plan=read(PLAN)
    train,tprov,tchecks=load_records(TRAIN,rows,P,plan)
    model=fit(train,P,basis);loo=[]
    for held in TRAIN:
        cm=fit([r for r in train if r['rank']!=held],P,basis)
        loo.extend(metric([r for r in train if r['rank']==held],cm,P)['networks'])
    tr=metric(train,model,P);ph=physical(train,model,P)
    frozen=dict(utc=utc(),hidden_readout=model['hidden'].tolist(),gnn_literal=f.encode(model['parameters']),
        training_projection=model['projection'].tolist(),coefficient=model['coefficient'].tolist(),
        feature_scale=model['feature_scale'].tolist(),training_projection_orthogonality=model['training_projection_orthogonality'],
        float32_export_max_abs_error=model['float32_export_max_abs_error'],train=tr,loo=loo,train_physical=ph,
        validation_truth_not_opened=True,train_provenance=tprov,source_sha256=sha(__file__))
    f.jwrite(a.out/'FROZEN_TRAIN_v1.json',frozen)
    valid,vprov,vchecks=load_records(VALID,rows,P,plan);va=metric(valid,model,P)
    loogood=all(x['u0']>0 for x in loo);lm=float(np.mean([x['relative_pct'] for x in loo])) if loogood else None
    passed=bool(loogood and lm<0 and sum(x['u1']<x['u0'] for x in loo)>=3 and va['positive_baselines'] and va['mean_pct']<=-.1 and va['lower']>=2)
    result=dict(utc=utc(),status='PROXY_GATE_PASS_NATIVE_REQUIRED' if passed else 'REJECT_BEFORE_NATIVE',passed=passed,
        train=tr,validation=va,loo=loo,loo_mean_pct=lm,train_physical=ph,
        source_sha256=sha(__file__),frozen_train_sha256=sha(a.out/'FROZEN_TRAIN_v1.json'),prereg_sha256=sha(a.out/'PREREG_v1.json'),
        provenance=tprov+vprov,gate_replay=tchecks+vchecks,captures=paths,
        training_projection_orthogonality=model['training_projection_orthogonality'],
        float32_export_max_abs_error=model['float32_export_max_abs_error'],replay={k:v for k,v in replay.items() if k.startswith('replay')},
        no_adjusted_gain_claim=True,reference_submission=337422)
    f.jwrite(a.out/'FITTED_v1.json',result)
    c=export(a.out,model,P,functions,sha(a.out/'FITTED_v1.json')) if passed else None
    print(json.dumps(dict(status=result['status'],train=tr,validation=va,loo_mean_pct=lm,train_physical=ph['total'],
        science=c['science'] if c else None,fit_sha256=sha(a.out/'FITTED_v1.json'))),flush=True)


if __name__=='__main__':main()
