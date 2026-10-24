from pathlib import Path
import json,hashlib,gc
import numpy as np
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for e in read(HERE/'analysis_seal.json')['sources']:assert pin(e['path'])==e
names=('zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus')
entries={}
for name in names:
    g=read(HERE/(name+'_guard.json'));assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    r=read(HERE/name/'complete.json');assert r['passive'] and r['source_inverse_exact'] and not r['final_targets_read']
    assert all(x['all16_bytes_exact'] and x['complete_bill_exact'] for x in r['rows'])
    entries[name]={x['mini']:x for x in r['rows']}
def load(name,mini):
    e=entries[name][mini]['packet'];assert pin(e['path'])==e
    with np.load(e['path']) as z:return {k:z[k].copy() for k in z.files}
def metrics(values):
    z,p,m,ph,mh=[np.asarray(x,dtype='float64') for x in values]
    rh=(p-m)/.25;rl=(ph-mh)/.125;den=float(np.linalg.norm(rl));pair=float(np.linalg.norm(p-m))
    return dict(secant_relative=float(np.linalg.norm(rh-rl)/den) if den else None,
        response_rms=float(np.sqrt(np.mean((.125*rl)**2))),
        curvature=float(np.linalg.norm(p+m-2*z)/pair) if pair else None)
def spectrum(Y):
    s=np.linalg.svd(Y.astype('float64'),compute_uv=False)
    return dict(largest=float(s[0]),smallest=float(s[-1]),condition=float(s[0]/s[-1]) if s[-1] else None,singular_values=s.tolist())
def eigen(C):
    v=np.linalg.eigvalsh((C.astype('float64')+C.astype('float64').T)*.5)
    scale=float(np.max(abs(v)));tol=1e-12*scale
    return dict(minimum=float(v[0]),maximum=float(v[-1]),negative_count=int(np.sum(v<0)),
        negative_beyond_relative_1e_minus12=int(np.sum(v < -tol)),eigenvalues=v.tolist())
def physical(Q,previous,C=None):
    X=np.zeros((1024,Q.shape[1]),dtype='float64');X[previous]=Q.astype('float64')
    return X@X.T if C is None else (X@C.astype('float64'))@X.T
def sketch(a,rows):
    J=a['J'].astype('float64');Qp=a['Qp'].astype('float64');core=a['core_in'].astype('float64')
    return J@(a['weight'].astype('float64')[:,None]*J[rows].T)+Qp@(core@Qp[rows].T)
def reference(Y,rows):
    Q,R=np.linalg.qr(Y,mode='reduced');K=Y[rows];K=(K+K.T)*.5
    C=R@np.linalg.solve(K,R.T);C=(C+C.T)*.5
    assert np.isfinite(C).all()
    return Q,R,K,C
def aligned_sketch(Y,previous,column_ids):
    order=np.argsort(column_ids,kind='stable');X=np.zeros((1024,Y.shape[1]),dtype='float64')
    X[previous]=Y[:,order]
    return X
rows=[];out=HERE/'reference';out.mkdir()
for mini in (57,68):
    data={name:load(name,mini) for name in names};z=data['zero']
    zero_ids=z['previous'][z['rows']]
    references={name:{} for name in names};aligned={name:{} for name in names};checks=[];packet={}
    selectors={};same_columns=True;all_fixed=True
    for name in names:
        a=data[name];previous=a['previous'];selected=a['rows'];ids=previous[selected]
        assert len(np.unique(previous))==len(previous) and len(np.unique(ids))==len(ids)
        same_set=np.array_equal(np.sort(ids),np.sort(zero_ids));same_columns &= same_set
        selectors[name]=dict(previous_same_order=np.array_equal(previous,z['previous']),
            previous_same_set=np.array_equal(np.sort(previous),np.sort(z['previous'])),
            selected_physical_same_order=np.array_equal(ids,zero_ids),selected_physical_same_set=same_set,
            selected_intersection=int(len(np.intersect1d(ids,zero_ids))),rank=len(ids))
        Q32,R32=np.linalg.qr(a['Y'],mode='reduced')
        assert Q32.tobytes()==a['Qn'].tobytes() and R32.tobytes()==a['R'].tobytes()
        K=a['Y'][selected].astype('float64');K=(K+K.T)*.5
        assert K.tobytes()==a['Ksym'].tobytes()
        R64=a['R'].astype('float64');core=(R64@np.linalg.solve(K,R64.T)).astype('float32');core=(core+core.T)*.5
        assert core.tobytes()==a['core_out'].tobytes()
        Y64=sketch(a,selected);Q64,R64,K64,core64=reference(Y64,selected)
        references[name]['native_projector']=physical(a['Qn'],previous)
        references[name]['reference_projector']=physical(Q64,previous)
        references[name]['native_represented_core']=physical(a['Qn'],previous,a['core_out'])
        references[name]['reference_represented_core']=physical(Q64,previous,core64)
        aligned[name]['native_sketch']=aligned_sketch(a['Y'],previous,ids)
        aligned[name]['reference_sketch']=aligned_sketch(Y64,previous,ids)
        lookup={int(v):i for i,v in enumerate(previous)}
        fixed_available=all(int(v) in lookup for v in zero_ids);all_fixed &= fixed_available
        fixed_check=None
        if fixed_available:
            fixed_rows=np.array([lookup[int(v)] for v in zero_ids],dtype='int64')
            Yfixed=sketch(a,fixed_rows);Qfixed,Rfixed,Kfixed,Cfixed=reference(Yfixed,fixed_rows)
            references[name]['fixed_projector']=physical(Qfixed,previous)
            references[name]['fixed_represented_core']=physical(Qfixed,previous,Cfixed)
            aligned[name]['fixed_sketch']=aligned_sketch(Yfixed,previous,zero_ids)
            fixed_check=dict(sketch_spectrum=spectrum(Yfixed),core_eigen=eigen(Cfixed))
        P=references[name]['native_projector'];Pr=references[name]['reference_projector']
        C=references[name]['native_represented_core'];Cr=references[name]['reference_represented_core']
        check=dict(name=name,qr_bytes_exact=True,core_bytes_exact=True,
            sketch_relative_error=float(np.linalg.norm(a['Y'].astype('float64')-Y64)/np.linalg.norm(Y64)),
            projector_relative_error=float(np.linalg.norm(P-Pr)/np.linalg.norm(Pr)),
            represented_core_relative_error=float(np.linalg.norm(C-Cr)/np.linalg.norm(Cr)),
            native_sketch_spectrum=spectrum(a['Y']),reference_sketch_spectrum=spectrum(Y64),
            native_K_eigen=eigen(K),reference_K_eigen=eigen(K64),old_core_eigen=eigen(a['core_in']),
            native_core_eigen=eigen(a['core_out']),reference_core_eigen=eigen(core64),
            fixed_rows_available=fixed_available,fixed_reference=fixed_check)
        checks.append(check)
        for key,val in references[name].items():packet[name+'_'+key+'_diag']=np.diag(val).copy()
        print(json.dumps(dict(event='nystrom_replayed',mini=mini,name=name,
            sketch_error=check['sketch_relative_error'],projector_error=check['projector_relative_error'],
            core_error=check['represented_core_relative_error'],native_condition=check['native_sketch_spectrum']['condition'],
            reference_condition=check['reference_sketch_spectrum']['condition'],selectors=selectors[name])),flush=True)
    keys=['native_projector','reference_projector','native_represented_core','reference_represented_core']
    if all_fixed:keys += ['fixed_projector','fixed_represented_core']
    reference_metrics={key:metrics([references[name][key] for name in names]) for key in keys}
    sketch_metrics={}
    if same_columns:
        for key in ('native_sketch','reference_sketch'):sketch_metrics[key]=metrics([aligned[name][key] for name in names])
    if all_fixed:sketch_metrics['fixed_sketch']=metrics([aligned[name]['fixed_sketch'] for name in names])
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,**packet)
    row=dict(mini=mini,selectors=selectors,replay_checks=checks,reference_metrics=reference_metrics,
        aligned_sketch_metrics=sketch_metrics,same_selected_physical_sets=same_columns,all_fixed_available=all_fixed,packet=pin(p))
    rows.append(row);print(json.dumps(dict(event='response_metrics',mini=mini,reference_metrics=reference_metrics,sketch_metrics=sketch_metrics)),flush=True)
    del data,references,aligned,packet;gc.collect()
once(HERE/'analysis.json',dict(rows=rows,final_targets_read=False,native_replacement=False,coefficient_fit=False,
    physical_coordinates_used=True,all_outputs_and_bills_exact=True))
