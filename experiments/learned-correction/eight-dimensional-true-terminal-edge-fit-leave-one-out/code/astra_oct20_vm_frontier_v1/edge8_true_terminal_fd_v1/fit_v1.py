"""One fixed true-terminal output8 fit; TRAIN4 only; no validation access."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import ast,hashlib,importlib.util,json,math,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace')
D=HERE.parent.parent/'astra_oct20_bill_v1'
TRAIN=[35,150,200,250]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
read=lambda p:json.loads(Path(p).read_text())
def write(path,x):
    with path.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def solve(JJ,rr,QQ):
    J=np.concatenate(JJ);r=np.concatenate(rr)
    c=np.maximum(np.sqrt(np.mean(J*J,axis=0)),1e-15)
    X=J/c
    G=X.T@X/len(X)+.1*np.eye(8)
    t=.5*np.linalg.solve(G,X.T@r/len(X))/c
    pred=J@t
    predicted_rms=float(np.sqrt(np.mean(pred*pred)))
    residual_rms=float(np.sqrt(np.mean(r*r)))
    qnorm=np.sqrt(np.mean(np.concatenate(QQ)**2,axis=0))
    curvature=float(np.sum(qnorm*t*t))
    radius=float(np.linalg.norm(t))
    alpha=min(1.,1/max(radius,1e-30),.25*residual_rms/max(predicted_rms,1e-30),
        .25*predicted_rms/max(curvature,1e-30))
    final=t*alpha
    shift=J@final
    final_rms=float(np.sqrt(np.mean(shift*shift)))
    final_curvature=float(np.sum(qnorm*final*final))
    assert np.linalg.norm(final)<=1+1e-10
    assert final_rms<=.25*residual_rms+1e-12
    assert final_curvature<=.25*final_rms+1e-12
    return final,dict(column_scale=c.tolist(),condition=float(np.linalg.cond(G)),
        damped_step_before_trust=t.tolist(),radial_multiplier=alpha,
        pretrust_fd_radius=radius,final_fd_radius=float(np.linalg.norm(final)),
        train_residual_normalized_rms=residual_rms,linear_shift_normalized_rms=final_rms,
        diagonal_curvature_bound_normalized_rms=final_curvature,
        curvature_ratio=final_curvature/max(final_rms,1e-30),ridge=.1,damping=.5)

def score(y,A,B):
    return float(np.mean((y-A)*(y-B)))
def result(rows):
    lr=np.array([math.log(x['U']/x['U0']) for x in rows])
    return dict(n=len(rows),raw_proxy_pct=float(100*np.expm1(lr.mean())),
        log_se_pct=float(100*lr.std(ddof=1)/math.sqrt(len(lr))) if len(lr)>1 else 0.,
        lower=int(np.sum(lr<0)))
def main():
    prereg=read(HERE/'PREREG_v1.json');build=read(HERE/'BUILD_v1.json')
    plan=read(HERE/'PLAN_v1.json');labels=read(HERE/'FIT_INPUTS_v1.json')
    assert sha(HERE/'PREREG_v1.json')==build['prereg_sha256']=='f448c6667d2cdd3085bbaddea3abb0cb21f235b913ff4ffb1cb748769aab81cf'
    assert sha(HERE/'PLAN_v1.json')==build['plan_sha256']=='8592f98e08e751287e9afcaca0b8de4fb239e423d14af39428228decc122077f'
    assert sha(HERE/'FIT_INPUTS_v1.json')==build['fit_inputs_sha256']
    assert set(plan['rows'])==set(labels['truth'])==set(map(str,TRAIN))
    assert all(set(x)=={'weights'} for x in plan['rows'].values())
    helper=D/'edge6_final_teacher_fd_v1/fit_v1.py'
    assert sha(helper)=='e1fd93d0f7fab2f853f6856c85ffb8ddc117633e5409deae535cd005d770ea11'
    spec=importlib.util.spec_from_file_location('checked_fd_records',helper)
    checked=importlib.util.module_from_spec(spec);spec.loader.exec_module(checked)
    evidence=[]
    base=checked.checked_train(prereg['baseline_tag'],build['source_sha256'],evidence,plan)
    arms={r['name']:checked.checked_train(r['name'],r['effective_sha256'],evidence,plan) for r in build['records']}
    data={};JJ=[];rr=[];QQ=[];curvature=[]
    for rank in TRAIN:
        truth=[]
        for key in ['truth_A','truth_B']:
            rec=labels['truth'][str(rank)][key];p=ROOT/rec['path']
            assert p.is_relative_to(ROOT) and sha(p)==rec['sha256']
            with np.load(p,allow_pickle=False) as z:
                assert z['mc_post'].shape==(16,1024)
                truth.append(z['mc_post'][15].astype('f8'))
        A,B=truth; y0=base[rank];U0=score(y0,A,B);assert U0>0
        scale=math.sqrt(U0)
        J=[];Q=[]
        for j in range(8):
            yp=arms[f'edge8_terminal_fd_c{j}_plus_v1'][rank]
            ym=arms[f'edge8_terminal_fd_c{j}_minus_v1'][rank]
            first=(yp-ym)*.5;second=(yp+ym-2*y0)*.5
            J.append(first);Q.append(second)
            curvature.append(dict(rank=rank,index=j,first_rms=float(np.sqrt(np.mean(first*first))),
                second_rms=float(np.sqrt(np.mean(second*second))),
                second_over_first=float(np.linalg.norm(second)/max(np.linalg.norm(first),1e-30))))
        J=np.stack(J,axis=1);Q=np.stack(Q,axis=1);residual=(A+B)*.5-y0
        data[rank]=dict(A=A,B=B,y0=y0,U0=U0,J=J)
        JJ.append(J/scale);QQ.append(Q/scale);rr.append(residual/scale)
    t,diag=solve(JJ,rr,QQ)
    train_rows=[];loo=[]
    for i,rank in enumerate(TRAIN):
        z=data[rank]
        y=np.maximum(z['y0']+z['J']@t,0).astype('f4').astype('f8')
        U=score(y,z['A'],z['B']);assert U>0
        train_rows.append(dict(rank=rank,U0=z['U0'],U=U))
        ids=[k for k in range(4) if k!=i]
        lt,ld=solve([JJ[k] for k in ids],[rr[k] for k in ids],[QQ[k] for k in ids])
        y=np.maximum(z['y0']+z['J']@lt,0).astype('f4').astype('f8')
        U=score(y,z['A'],z['B']);assert U>0
        loo.append(dict(rank=rank,U0=z['U0'],U=U,fd_coordinate_step=lt.tolist(),diagnostic=ld))
    tr=result(train_rows);cv=result(loo)
    passed=tr['raw_proxy_pct']<=-.25 and cv['raw_proxy_pct']<0 and cv['lower']>=3
    original=np.array(prereg['original_output'],dtype='f8')
    scales=np.array(prereg['coordinate_scales'],dtype='f8')
    output=(original+prereg['h']*scales*t).astype('f4').astype('f8').tolist()
    frozen=dict(utc=utc(),status='FROZEN_TRAIN_ONLY_PASS' if passed else 'CLOSED_FIXED_TRAIN_LOO_GATE',
        source_sha256=build['source_sha256'],plan_sha256=build['plan_sha256'],
        prereg_sha256=build['prereg_sha256'],script_sha256=sha(__file__),train=TRAIN,
        validation=[259,277,287],validation_truth_or_predictions_opened=False,
        no_held25_or_fresh_read=True,no_accuracy_gain_claim=True,
        fd_coordinate_step=t.tolist(),original_output=original.tolist(),output=output,
        train_linear_proxy=tr,train_rows=train_rows,loo_proxy=cv,loo_rows=loo,
        diagnostic=diag,curvature=curvature,evidence=evidence,fit_truth_pins=labels['truth'])
    # Freeze once before any validation or source promotion.
    write(HERE/'FROZEN_TRAIN_FIT_v1.json',frozen)
    if passed:
        outdir=HERE/'native_v1';outdir.mkdir(exist_ok=False)
        tail=('# Frozen TRAIN4-only output8 true-terminal repair.\n'
            +'assert _EDGE_PARAMETERS is None and not _E2_EDGE_STATE\n'
            +'assert _EDGE_LITERAL[\'output\'] == '+repr(original.tolist())+'\n'
            +'_EDGE_LITERAL[\'output\'] = '+repr(output)+'\n'
            +'_FAST_RESCUE_FRAC = {}\n')
        parentdir=D/'edge_hidden_nonlinear_v1/native_hidden8_v1'
        sources={'science':('candidate_msg8_edge_hidden8_nonlinear_science_v1.py',build['source_sha256']),
                 'release':('candidate_msg8_edge_hidden8_nonlinear_release_v1.py','786c135a04ebf5b902ff533e0190b748c46f297282ae508ebcc2daa234b77180')}
        exports={}
        for role,(name,digest) in sources.items():
            parent=parentdir/name;assert sha(parent)==digest
            p=outdir/f'candidate_edge8_true_terminal_{role}_v1.py'
            with p.open('x') as f:f.write(parent.read_text()+'\n'+tail)
            assert ast.unparse(ast.parse(p.read_text()).body[-1])=='_FAST_RESCUE_FRAC = {}'
            exports[role]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),effective_sha256=sha(p),
                bytes=p.stat().st_size,parent_sha256=digest)
        write(HERE/'EXPORT_v1.json',dict(utc=utc(),outputs=exports,fit_sha256=sha(HERE/'FROZEN_TRAIN_FIT_v1.json'),
            successful=True,qualification_pending=True,no_native_validation_launched=True))
    print(json.dumps(dict(utc=utc(),status=frozen['status'],train=tr,loo=cv,diagnostic=diag,fit_sha256=sha(HERE/'FROZEN_TRAIN_FIT_v1.json'))))
if __name__=='__main__':main()

