from pathlib import Path
import hashlib,json,math,gc
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'recurrent_covariance_observer_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for e in read(HERE/'replay_seal.json')['sources']:assert pin(Path(e['path']))==e
names=('zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus')
entries={}
for name in names:
    g=read(HERE/(name+'_guard.json'));assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    r=read(HERE/name/'complete.json');assert r['passive'] and r['source_inverse_exact'] and not r['final_targets_read']
    entries[name]={x['mini']:x for x in r['rows']}
    assert all(x['all16_bytes_exact'] and x['complete_bill_exact'] for x in r['rows'])
base={x['mini']:x for x in read(BASE/'capture/complete.json')['rows']}
def load(name,mini):
    e=entries[name][mini]['packet'];assert pin(Path(e['path']))==e
    with np.load(e['path']) as z:return {k:z[k].copy() for k in z.files}
def metrics(values):
    z,p,m,ph,mh=[np.asarray(x,dtype='float64') for x in values];h=.125
    rh=(p-m)/(2*h);rl=(ph-mh)/h
    denom=float(np.linalg.norm(rl));pair=float(np.linalg.norm(p-m))
    return dict(secant_relative=float(np.linalg.norm(rh-rl)/denom) if denom else 0.,
        response_rms=float(np.sqrt(np.mean((h*rl)**2))),
        curvature=float(np.linalg.norm(p+m-2*z)/pair) if pair else 0.,
        nonlinear_remainder_rms=float(np.sqrt(np.mean((p+m-2*z)**2))),
        all_bytes_equal=all(x.tobytes()==z.tobytes() for x in (p,m,ph,mh)))
def transport(w,Q,B,wide=False):
    wt=w.T;wt2=wt*wt
    if wide:P1=wt.astype('float64')@Q;T2=wt2.astype('float64')@B.T
    else:P1=wt@Q;T2=wt2@B.T
    sub=(T2@P1.T)/3
    diagonal=np.sum(P1*T2,axis=1,dtype='float64')
    result=sub.astype('float64');np.fill_diagonal(result,diagonal)
    return result
def factor(A,cols,dtype):
    X=A.astype(dtype);Z=X[:,cols];s0=np.linalg.svd(Z,compute_uv=False)
    Q=np.linalg.qr(Z,mode='reduced')[0]
    Q=np.linalg.qr(X@np.linalg.qr(X.T@Q,mode='reduced')[0],mode='reduced')[0]
    B=Q.T@X;U,S,Vt=np.linalg.svd(B,full_matrices=False)
    return Q@U[:,:32],S[:32,None]*Vt[:32],dict(initial_condition=float(s0[0]/s0[-1]),
        retained_boundary_relative_gap=float((S[31]-S[32])/S[31]),singular_values=S.astype('float64').tolist())
continuous=('C8','star8','Cpre9','star_pre9','CPpre9','K21_9','k4_9','physical_mean9','mean_delta9','post_mean9','features9','hidden9','sigma9')
scalars=('b8','radial8','birth8','u8','delta8','t8','s8','previous_hidden8','mu9','bpre9')
selectors=('selected8','alive8','live9','response9','star_columns9')
rows=[];out=HERE/'replay';out.mkdir()
for mini in (57,68):
    a={name:load(name,mini) for name in names};z=a['zero'];wentry=base[mini]['weights']
    assert pin(Path(wentry['path']))==wentry
    w=np.load(wentry['path'],mmap_mode='r')[9].copy().astype('float32')
    constant={key:all(x[key].tobytes()==z[key].tobytes() for x in a.values()) for key in scalars}
    selection={key:{name:dict(same_order=x[key].tobytes()==z[key].tobytes(),same_set=np.array_equal(np.sort(x[key].ravel()),np.sort(z[key].ravel()))) for name,x in a.items()} for key in selectors}
    fields={key:metrics([a[name][key] for name in names]) for key in continuous}
    for key in ('C8','Cpre9','star_pre9','CPpre9','K21_9'):
        fields[key+'_diag']=metrics([np.diag(a[name][key]) for name in names])
    replay={name:{} for name in names};checks=[];packet={}
    for name in names:
        x=a[name];A=x['star8'];cols=x['star_columns9'];Q=x['star_factor_Q9'];B=x['star_factor_B9']
        exact=transport(w,Q,B);native=x['star_pre9']
        rel=float(np.linalg.norm(exact-native)/max(np.linalg.norm(native),1e-30))
        drel=float(np.linalg.norm(np.diag(exact-native))/max(np.linalg.norm(np.diag(native)),1e-30))
        assert rel<=2e-6 and drel<=2e-6,(mini,name,rel,drel)
        Q32,B32,s32=factor(A,cols,'float32');r32=transport(w,Q32,B32)
        Q64,B64,s64=factor(A,cols,'float64')
        r64cast=transport(w,Q64.astype('float32'),B64.astype('float32'))
        r64=transport(w,Q64,B64,True)
        wt=w.T.astype('float64');wt2=(w.T*w.T).astype('float64')
        full=wt2@A.astype('float64').T@w.astype('float64');full_diag=np.diag(full).copy();full/=3;np.fill_diagonal(full,full_diag)
        represented=Q.astype('float64')@B.astype('float64')
        rep64=Q64@B64
        for key,val in [('represented_native',represented),('represented64',rep64),('fp32_replay',r32),('factor64_cast32',r64cast),('factor64_full64',r64),('full64_linear',full)]:
            replay[name][key]=val
            packet[f'{name}_{key}_diag']=np.diag(val).copy()
        checks.append(dict(name=name,captured_factor_transport_relative=rel,captured_factor_diagonal_relative=drel,
            fp32_replay_transport_relative=float(np.linalg.norm(r32-native)/np.linalg.norm(native)),
            fp32_replay_diagonal_relative=float(np.linalg.norm(np.diag(r32-native))/np.linalg.norm(np.diag(native))),
            factor64_change_relative=float(np.linalg.norm(r64cast-native)/np.linalg.norm(native)),
            factor64_diagonal_change_relative=float(np.linalg.norm(np.diag(r64cast-native))/np.linalg.norm(np.diag(native))),
            fp32_spectrum=s32,fp64_spectrum=s64))
        # Verify the source-level post8 trace relation up to the stored-star FP32 rounding.
        change=-6*(x['C8']-z['C8']).T*z['b8'][None,:]
        err=(A.astype('float64')-z['star8'].astype('float64'))-change
        bound=2*(abs(np.spacing(A))+abs(np.spacing(z['star8']))).astype('float64')+1e-14
        assert np.all(abs(err)<=bound),(mini,name,float(np.max(abs(err)/bound)))
        checks[-1]['post8_star_relation_max_fraction_bound']=float(np.max(abs(err)/bound))
        print(json.dumps(dict(event='replayed',mini=mini,name=name,native_transport_relative=rel,fp32_replay_relative=checks[-1]['fp32_replay_transport_relative'],factor64_relative=checks[-1]['factor64_change_relative'])),flush=True)
    reference_metrics={}
    for key in replay['zero']:
        reference_metrics[key]=dict(full=metrics([replay[name][key] for name in names]),diagonal=metrics([np.diag(replay[name][key]) for name in names]))
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,**packet)
    rows.append(dict(mini=mini,constant_state=constant,selectors=selection,fields=fields,
        reference_metrics=reference_metrics,replay_checks=checks,packet=pin(p)))
    del a,replay,packet;gc.collect()
once(HERE/'analysis.json',dict(rows=rows,final_targets_read=False,source_repair=False,
    native_replacement=False,coefficient_fit=False,all_trace_outputs_and_bills_exact=True))
print(json.dumps(dict(summary=[dict(mini=r['mini'],constant=r['constant_state'],native_star_diag=r['fields']['star_pre9_diag'],references={k:v['diagonal'] for k,v in r['reference_metrics'].items()},physical_mean=r['fields']['physical_mean9'],head=r['fields']['mean_delta9']) for r in rows])),flush=True)
