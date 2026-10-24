"""Frozen TRAIN4 post-message hidden128 variance predictability screen.

No source export, evaluation-only truth, layer selection, or native launch.
The remaining mean-innovation check rejects double-counted head corrections.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
V=HERE.parent
ROOT=Path('/home/arcworker/arc-workspace')
RUNS=Path('/home/arcworker/arc-runs')
TRAIN=(35,150,200,250)
RIDGE=.1
DAMPING=.5
sys.path.insert(0,str(V))
import fit_frontier_v1 as f
import fit_current_msg8_messages_v2 as current
import fit_hidden_complement_innovation_v1 as innovation
sha=f.sha
read=lambda p:json.loads(Path(p).read_text())

def fit(records):
    H=np.concatenate([r['hidden2'] for r in records])
    y=np.concatenate([.5*(r['variance_A']+r['variance_B']) for r in records])
    center=H.mean(axis=0)
    scale=np.maximum(np.sqrt(np.mean((H-center)**2,axis=0)),1e-8)
    Z=np.column_stack(((H-center)/scale,np.ones(len(H))))
    c=np.linalg.solve(Z.T@Z+RIDGE*len(Z)*np.eye(129),Z.T@y)
    w=(c[:128]/scale).astype('f4')
    b=np.float32(c[128]-center@(c[:128]/scale))
    assert np.isfinite(w).all() and np.isfinite(b)
    return dict(weight=w,bias=b,feature_center=center,feature_scale=scale)

def predict(r,model):
    return r['hidden2']@model['weight'].astype('f8')+float(model['bias'])

def metric(records,model):
    rows=[]
    for rank in sorted({r['rank'] for r in records}):
        rr=[r for r in records if r['rank']==rank]
        va=np.concatenate([r['variance_A'] for r in rr]);vb=np.concatenate([r['variance_B'] for r in rr])
        vp=np.concatenate([predict(r,model) for r in rr])
        ia=np.concatenate([r['A'] for r in rr]);ib=np.concatenate([r['B'] for r in rr])
        # .5*pdf/sigma times predicted prevariance error, damped once.
        shift=np.concatenate([DAMPING*.5*r['sigma']*r['pdf']*predict(r,model)*r['mask']/r['physical'] for r in rr])
        uvar=float(np.mean(va*vb));nvar=float(np.mean((va-vp)*(vb-vp)))
        umean=float(np.mean(ia*ib));nmean=float(np.mean((ia-shift)*(ib-shift)))
        assert uvar>0 and umean>0
        physical=np.concatenate([r['physical'] for r in rr])
        saved=np.concatenate([r['saved_delta'] for r in rr])
        oracle=np.concatenate([DAMPING*.5*r['sigma']*r['pdf']*.5*(r['variance_A']+r['variance_B'])*r['mask']/r['physical'] for r in rr])
        rows.append(dict(rank=rank,variance_U0=uvar,variance_U1=nvar,variance_r2=1-nvar/uvar,
            innovation_U0=umean,innovation_U1=nmean,innovation_relative_pct=100*(nmean/umean-1),
            oracle_variance_innovation_pct=100*(float(np.mean((ia-oracle)*(ib-oracle)))/umean-1),
            physical_correction_rms=float(np.sqrt(np.mean((shift*physical)**2))),
            physical_correction_max_abs=float(np.max(np.abs(shift*physical))),
            old_head_rms=float(np.sqrt(np.mean(saved*saved))),
            correction_innovation_cos=float(np.dot(shift,.5*(ia+ib))/max(np.linalg.norm(shift)*np.linalg.norm(.5*(ia+ib)),1e-30))))
    return dict(networks=rows,variance_r2_mean=float(np.mean([r['variance_r2'] for r in rows])),
        variance_positive=sum(r['variance_r2']>0 for r in rows),
        innovation_mean_pct=float(np.mean([r['innovation_relative_pct'] for r in rows])),
        innovation_lower=sum(r['innovation_U1']<r['innovation_U0'] for r in rows))

def selftest():
    rng=np.random.default_rng(2026102108)
    H=rng.normal(size=(1600,128));c=rng.normal(size=128)*.01;y=H@c+.04
    r=dict(hidden2=H,variance_A=y,variance_B=y)
    model=fit([r]);got=predict(r,model)
    assert np.mean((got-y)**2)<.04*np.mean(y*y)
    mu=np.array([-.7,.2,1.1]);var=np.array([.4,.8,1.3]);eps=1e-6
    def relu(v):
        sig=np.sqrt(v);a=mu/sig;gate=np.array([.5*(1+math.erf(t/math.sqrt(2))) for t in a])
        return sig*np.exp(-a*a/2)/math.sqrt(2*math.pi)+mu*gate
    fd=(relu(var+eps)-relu(var-eps))/(2*eps)
    expected=np.exp(-mu*mu/(2*var))/math.sqrt(2*math.pi)/(2*np.sqrt(var))
    np.testing.assert_allclose(fd,expected,rtol=2e-8,atol=1e-10)
    return dict(ridge_recovery=True,variance_to_relu_mean_derivative=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    checks=selftest()
    if args.self_test:print(json.dumps(checks));return
    assert sys.platform=='linux'
    for p,pin in [(f.__file__,'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'),
                  (current.__file__,'56d3c63e1d766e6b1870e6aa07b88b84c02172d71edf28a6ad6a1724ad3f53ea'),
                  (innovation.__file__,'884b6183f05a0c31cf58aa204eb3907259a4ebc3b19982af9b0d8d897fe0fb7c')]:assert sha(p)==pin
    prereg=read(HERE/'PREREG_v1.json')
    assert prereg['train']==list(TRAIN) and prereg['ridge']==RIDGE and prereg['damping']==DAMPING
    assert prereg['script_sha256']==sha(__file__)
    planp=V/'PLAN_BANK8_ACTUAL_SHARED_v1.json'
    assert sha(planp)=='cf76262261a85fcb6b5c508ad37a2b13f92884b6b69aff66bd0da4bcd0833793'
    plan=read(planp)
    source=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py'
    assert sha(source)==current.SCI
    literals,_=f.literals_and_functions(source.read_text());P=f.decode(literals['_GNN_LITERAL'])
    raw,paths,meta=current.collect(RUNS/'current_msg8_gnn_capture_v1')
    captured=f.decode(meta['gnn_literal']);assert all(np.array_equal(v,captured[k]) for k,v in P.items())
    raw=[r for r in raw if r['rank'] in TRAIN]
    assert len(raw)==60
    replay=current.matrices(raw,P)
    records,provenance,head_checks=innovation.load_records(TRAIN,raw,P,plan)
    selected={(r['rank'],r['layer']):r for r in raw}
    truth={};truth_pins=[]
    for rank in TRAIN:
        halves={}
        for half in ('A','B'):
            item=plan['rows'][str(rank)]['truth_'+half];path=ROOT/item['path']
            assert sha(path)==item['sha256']
            with np.load(path,allow_pickle=False) as z:
                assert int(z['rank'])==rank
                halves[half]=z['mc_pre_v'].astype('f8')
                assert halves[half].shape==(16,1024) and np.isfinite(halves[half]).all()
            truth_pins.append(dict(rank=rank,half=half,path=str(path),sha256=sha(path),target_key='mc_pre_v'))
        truth[rank]=halves
    feature_checks=[]
    for r in records:
        rr=selected[(r['rank'],r['layer'])];ids=np.asarray(rr['rows']);l=r['layer']
        H=f.array(rr,'H').astype('f8');sig=f.array(rr,'sigma').astype('f8')
        assert np.max(np.abs(H[:,[3,6]]))<7.999,'Cannot reconstruct clipped alpha/pdf feature'
        alpha=H[:,3]*float(P['scales'][l,3])+float(P['centers'][l,3])
        pdf=H[:,6]*float(P['scales'][l,6])+float(P['centers'][l,6])
        oracle=np.exp(-.5*alpha*alpha)/math.sqrt(2*math.pi)
        error=float(np.max(np.abs(pdf-oracle)));assert error<2e-6
        assert sig.min()>0 and pdf.min()>=0
        r.update(sigma=sig,pdf=pdf)
        for half in ('A','B'):r['variance_'+half]=(truth[r['rank']][half][l,ids]-sig*sig)/(sig*sig)
        feature_checks.append(dict(rank=r['rank'],layer=l,pdf_replay_max_abs=error))
    model=fit(records);train=metric(records,model);loo=[]
    for held in TRAIN:
        m=fit([r for r in records if r['rank']!=held])
        loo.extend(metric([r for r in records if r['rank']==held],m)['networks'])
    varmean=float(np.mean([r['variance_r2'] for r in loo]));meanmean=float(np.mean([r['innovation_relative_pct'] for r in loo]))
    varlower=sum(r['variance_r2']>0 for r in loo);meanlower=sum(r['innovation_U1']<r['innovation_U0'] for r in loo)
    passed=bool(train['variance_r2_mean']>=.05 and varmean>=.02 and varlower>=3 and meanmean<=-.15 and meanlower>=3)
    result=dict(utc=f.utc(),status='TRAIN_SCREEN_PASS_PRICE_REQUIRED' if passed else 'CLOSED_TRAIN_VARIANCE_OR_MEAN_ALIGNMENT',
        passed=passed,train=train,loo=loo,loo_variance_r2_mean=varmean,loo_innovation_mean_pct=meanmean,
        loo_variance_positive=varlower,loo_innovation_lower=meanlower,
        weight=model['weight'].tolist(),bias=float(model['bias']),feature_center=model['feature_center'].tolist(),
        feature_scale=model['feature_scale'].tolist(),prereg_sha256=sha(HERE/'PREREG_v1.json'),script_sha256=sha(__file__),
        source_sha256=current.SCI,plan_sha256=sha(planp),provenance=provenance,truth_pins=truth_pins,
        target_free_capture_files=[p for p in paths if any('/rank%04d_'%r in p['path'] for r in TRAIN)],
        replay={k:v for k,v in replay.items() if k.startswith('replay')},head_checks=head_checks,feature_checks=feature_checks,
        selftests=checks,all_truth_ranks_read=list(TRAIN),no_validation_held25_or_fresh_truth=True,
        no_candidate_exported=True,no_adjusted_gain_claim=True,added_bill_unpriced=True)
    f.jwrite(HERE/'SCREEN_v1.json',result)
    print(json.dumps({k:result[k] for k in ['utc','status','train','loo_variance_r2_mean','loo_innovation_mean_pct','loo_variance_positive','loo_innovation_lower']}))

if __name__=='__main__':main()
