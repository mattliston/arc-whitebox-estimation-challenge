"""Frozen per-layer hidden2/4 covariance-edge distillation, from actual777a captures."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import subprocess
import time
import numpy as np
HERE=Path(__file__).resolve().parent;D=HERE.parent;RUNS=Path('/home/arcworker/arc-runs')
PARENT=D/'edge_hidden6_nonlinear_v1/fit_edge_hidden6_v1.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(PARENT)=='447e3058b972c65459a3e04f302d9e194ee6502788d6a792fe93a997b30ac36f'
spec=importlib.util.spec_from_file_location('physical_edge_fit',PARENT)
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
read=lambda p:json.loads(Path(p).read_text())
WIDTHS=(2,4);STEPS=5000;WORKERS=4
RECORDS=None;META=None;RIDGE=None
def write(p,x):
    with Path(p).open('x') as s:json.dump(x,s,indent=2,allow_nan=False);s.write('\n')
def fit_one(task):
    width,layer=task;f.WIDTH=width
    tr=[r for r in RECORDS if r['rank'] in f.TRAIN and r['layer']==layer]
    va=[r for r in RECORDS if r['rank'] in f.VALID and r['layer']==layer]
    assert len(tr)==4 and len(va)==3
    aa,bb=(float(np.float32(META['activation'][k])) for k in ('a','b'))
    G=sum(r['G'] for r in tr)/len(tr);original=tr[0]['output']
    init=RIDGE.fit_width(G,original,width,1e-6)
    anchor=np.concatenate((np.eye(14)[:,init['keep']].ravel(),np.zeros(width),np.asarray(init['output'])))
    v=anchor.copy();data=f.pool(tr);N=len(data['target'])
    best=v.copy();bestloss=f.objective(v,data,aa,bb,anchor,False)
    first=bestloss;m=np.zeros_like(v);s=np.zeros_like(v)
    rng=np.random.default_rng(20261021+100*width+layer)
    history=[];start=time.monotonic()
    for step in range(1,STEPS+1):
        ix=rng.integers(0,N,size=min(8192,N));batch={k:val[ix] for k,val in data.items()}
        _,g=f.objective(v,batch,aa,bb,anchor,factor=N/len(ix))
        norm=float(np.linalg.norm(g))
        if norm>10:g*=10/norm
        m=.9*m+.1*g;s=.999*s+.001*g*g
        v-=.001*(m/(1-.9**step))/(np.sqrt(s/(1-.999**step))+1e-8)
        if step%250==0 or step==1:
            loss=f.objective(v,data,aa,bb,anchor,False)
            if loss<bestloss:best,bestloss=v.copy(),loss
            history.append(dict(step=step,train_objective=loss,best=bestloss))
    best=best.astype(np.float32).astype(np.float64);Q,c,o=f.unpack(best)
    edge={k:np.asarray(x,dtype=np.float32) if isinstance(x,list) else x for k,x in META['edge_literal'].items()}
    folded={}
    for key,value in edge.items():
        if key in ('left','right','edge'):folded[key]=(value@Q.astype(np.float32)).astype(np.float32).tolist()
        elif key=='bias':folded[key]=(value@Q.astype(np.float32)+c.astype(np.float32)).astype(np.float32).tolist()
        elif key=='output':folded[key]=o.astype(np.float32).tolist()
        else:folded[key]=value.tolist() if isinstance(value,np.ndarray) else value
    context={key:(np.asarray(value,dtype=np.float32)@Q.astype(np.float32)).astype(np.float32).tolist()
        for key,value in META['message_edge_literal'].items()}
    result=dict(width=width,layer=layer,utc=utc(),wall_s=time.monotonic()-start,initialization=init,
        Q=Q.tolist(),offset=c.tolist(),output=o.tolist(),edge_literal=folded,message_edge_literal=context,
        train_before=f.metrics(anchor,tr,aa,bb),train_after=f.metrics(best,tr,aa,bb),
        validation_before=f.metrics(anchor,va,aa,bb),validation_after=f.metrics(best,va,aa,bb),
        history=history,initial_train_objective=first,best_train_objective=bestloss,
        uses_truth=False,validation_selection=False)
    return result
def main():
    global RECORDS,META,RIDGE
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    checks=[]
    for width in WIDTHS:
        f.WIDTH=width;checks.append(dict(width=width,**f.selftest()))
    if a.self_test:print(json.dumps(checks));return
    assert os.uname().sysname=='Linux'
    verified=read(RUNS/'EDGE_PREPOST_CAPTURE_VERIFIED_v1.json')
    assert verified['successful'] and verified['no_truth']
    assert verified['teacher_sha256']==f.TEACHER_SHA and verified['train']==list(f.TRAIN) and verified['validation']==list(f.VALID)
    assert sha(RUNS/'edge_prepost_msg8_capture_v1/COMPLETE.json')==verified['complete_sha256']
    for item in verified['guards']:
        assert sha(item['path'])==item['sha256']
        g=read(item['path']);assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors'] and g['max_owned_swap_bytes']==0
    out=RUNS/'edge_layer_hidden2_4_fit_v1';out.mkdir(exist_ok=False)
    write(out/'PREREG_v1.json',dict(utc=utc(),widths=WIDTHS,layers=list(range(8,15)),steps=STEPS,workers=WORKERS,
        train=list(f.TRAIN),validation=list(f.VALID),learning_rate=.001,batch=8192,
        seed_rule='20261021+100*width+layer',objective='same physical covariance relative MSE and anchors as gradient-checked hidden6; fit layers independently',
        checkpoint_selection='training objective only every250; validation descriptive only',
        reason='shared feature map may limit compression; output-only repair already failed; fit the entire hidden map per layer',
        script_sha256=sha(__file__),parent_fitter_sha256=sha(PARENT),capture_verification_sha256=sha(RUNS/'EDGE_PREPOST_CAPTURE_VERIFIED_v1.json'),
        selftests=checks,no_truth=True,no_native_launch=True))
    RECORDS,META,provenance=f.collect(RUNS/'edge_prepost_msg8_capture_v1')
    rp=D/'cloud_edge_teacher_v1/fit_edge_output_v1.py'
    sp=importlib.util.spec_from_file_location('edge_ridge',rp);RIDGE=importlib.util.module_from_spec(sp);sp.loader.exec_module(RIDGE)
    results=[]
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=multiprocessing.get_context('fork')) as pool:
        for row in pool.map(fit_one,[(width,layer) for width in WIDTHS for layer in range(8,15)]):
            write(out/('width%d_layer%d.json'%(row['width'],row['layer'])),row);results.append(row)
            print(json.dumps(dict(utc=utc(),width=row['width'],layer=row['layer'],wall_s=row['wall_s'],
                train=row['train_after']['mean_relative_mse'],valid=row['validation_after']['mean_relative_mse'])),flush=True)
    aggregate={str(w):{split:float(np.mean([r[split]['mean_relative_mse'] for r in results if r['width']==w]))
        for split in ('train_before','train_after','validation_before','validation_after')} for w in WIDTHS}
    write(out/'FITTED_v1.json',dict(utc=utc(),widths=WIDTHS,train=list(f.TRAIN),validation=list(f.VALID),uses_truth=False,
        validation_selection=False,teacher_source_sha256=f.TEACHER_SHA,activation=META['activation'],recipe=META['recipe'],
        captures=provenance,results=results,aggregate=aggregate,script_sha256=sha(__file__),no_native_gain_claim=True,no_native_launch=True))
    print(json.dumps(dict(utc=utc(),aggregate=aggregate)),flush=True)
if __name__=='__main__':main()

