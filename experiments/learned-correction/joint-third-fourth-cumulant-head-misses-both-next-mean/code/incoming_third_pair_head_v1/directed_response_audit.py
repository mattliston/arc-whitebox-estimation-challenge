import ast,math,resource,time
from common import *
import numpy as np
from threadpoolctl import threadpool_info
from head import predict_s
verify();verify('fit_sources_sealed.json');verify('directed_audit_sealed.json')
assert all(x['num_threads']==1 for x in threadpool_info());start=time.monotonic()

def coefficients(C,s,a,p,g):
    rho=C/np.outer(s,s);np.fill_diagonal(rho,0)
    A=[g,p,-a*p,(a*a-1)*p];left=np.zeros_like(C);right=np.zeros_like(C)
    for r in range(3):
        rr=rho**r/math.factorial(r)
        left+=rr*np.outer(A[r+1],A[r]);right+=rr*np.outer(A[r],A[r+1])
    factor=s*s/np.diag(C);left*=np.outer(s,s)/2*factor[:,None];right*=np.outer(s,s)/2*factor[None,:]
    np.fill_diagonal(left,0);np.fill_diagonal(right,0)
    return left,right,rho

source=SOURCE.read_text();tree=ast.parse(source);node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pair_series_fused'][-1]
code=ast.get_source_segment(source,node);scope={'fnp':np};exec(compile(code,str(SOURCE),'exec'),scope)
rng=np.random.default_rng(219057);fixtures=[]
for asymmetric,floor in ((False,False),(True,False),(False,True),(True,True)):
    n=9;Q=rng.normal(0,.1,(n,n));C=Q@Q.T+np.eye(n)
    if asymmetric:C+=rng.normal(0,.015,(n,n))*(1-np.eye(n))
    if floor:C[0,0]=1e-12;C[0,1:]*=1e-6;C[1:,0]*=1e-6
    s=np.sqrt(np.maximum(np.diag(C),1e-10));a=rng.normal(0,.7,n)
    p=np.exp(-.5*a*a)/math.sqrt(2*math.pi);g=np.array([.5*math.erfc(-float(x)/math.sqrt(2)) for x in a])
    left,right,rho=coefficients(C,s,a,p,g);D=rng.normal(0,.01,(n,n));np.fill_diagonal(D,0)
    jets=np.zeros((7,n));jets[1]=g;jets[2]=p;jets[3]=-a*p;jets[4]=(a*a-1)*p
    F=np.stack([jets]*3);G=F.copy();Z=np.zeros((n,n));z=np.zeros(n)
    P=D*(s*s/np.diag(C))[:,None]
    baseline=scope['_pair_series_fused'](F,G,rho,Z,Z,Z,z,z,n)[0]
    changed=scope['_pair_series_fused'](F,G,rho,P,Z,Z,z,z,n)[0]
    direct=(changed-baseline)*np.outer(s,s);np.fill_diagonal(direct,0)
    actual=left*D+right*D.T;error=float(np.linalg.norm(actual-direct)/np.linalg.norm(direct));assert error<=2e-11,error
    q=left*D;surrogate=q+q.T;old_error=float(np.linalg.norm(surrogate-direct)/np.linalg.norm(direct))
    if asymmetric:assert old_error>1e-8,old_error
    fixtures.append(dict(asymmetric=asymmetric,active_floor=floor,exact_relative=error,symmetric_surrogate_relative=old_error))

asymmetries=[]
for ordinal in TRAIN+HELDOUT:
    cr=read(HERE/f'capture_mini{ordinal:04d}.json');check(cr['capture'])
    with np.load(cr['capture']['path'],allow_pickle=False) as z:
        for l in LAYERS:
            C=z[f'C_{l:02d}'];asymmetries.append(dict(ordinal=ordinal,layer=l,relative=float(np.linalg.norm(C-C.T)/np.linalg.norm(C)),minimum_variance=float(np.min(np.diag(C)))))

r=read(HERE/'information_result.json');f=read(HERE/'model_frozen.json')
for k in ('model','normalizers','training'):check(f[k])
t=dict(np.load(f['model']['path'],allow_pickle=False));energy=np.load(f['normalizers']['path'],allow_pickle=False)['energies'];records=[]
for ordinal in HELDOUT:
    ir=read(HERE/f'inputs_mini{ordinal:04d}.json');tr=read(HERE/f'targets_mini{ordinal:04d}.json');cr=read(HERE/f'capture_mini{ordinal:04d}.json')
    original=next(x for x in r['records'] if x['ordinal']==ordinal);rows=[];normalized=[]
    with np.load(cr['capture']['path'],allow_pickle=False) as z:
        for ri,rt in zip(ir['records'],tr['targets']):
            l=ri['layer'];check(ri['input']);check(rt['target']);c=dict(np.load(ri['input']['path'],allow_pickle=False));target=np.load(rt['target']['path'],allow_pickle=False)
            pred=predict_s(np,c['F'],c['R'],c['S'],t).astype(np.float64)
            left,right,_=coefficients(z[f'C_{l:02d}'],c['sigma'],z[f'alpha_{l:02d}'],z[f'pdf_{l:02d}'],z[f'gate_{l:02d}'])
            weight=left*left+right.T*right.T;weight/=np.sum(weight)/(1024*1023);terms=[];gaps=[]
            for e in (-target,pred-target):
                cov=left*e+right*e.T;response=np.sum(c['Wnext']*(cov@c['Wnext']),axis=0)*c['gnext']
                q=c['L']*e;old=q+q.T;oldresp=np.sum(c['Wnext']*(old@c['Wnext']),axis=0)*c['gnext']
                gaps.append([float(np.linalg.norm(cov-old)/np.linalg.norm(cov)),float(np.linalg.norm(response-oldresp)/np.linalg.norm(response))])
                terms.append(np.array([np.sum(weight*e*e)/(1024*1023),np.sum(cov*cov)/(1024*1023),np.mean(response*response)]))
            terms=np.array(terms);normalized.append(terms/energy[l-8]);rows.append(dict(layer=l,directed_ratios=(terms[1]/terms[0]).tolist(),surrogate_field_relative_gaps=gaps))
    pooled=np.sum(normalized,axis=0);ratios=pooled[1]/pooled[0];difference=ratios-np.array(original['ratios'])
    record=dict(ordinal=ordinal,original_ratios=original['ratios'],directed_ratios=ratios.tolist(),ratio_differences=difference.tolist(),layers=rows,
                original_primary_passed=original['passed'],directed_sensitivity_would_pass=bool(np.all(ratios<=[.85,.85,.90])))
    records.append(record);print(record,flush=True)
write(HERE/'directed_response_audit.json',dict(passed=True,fixtures=fixtures,asymmetries=asymmetries,records=records,
      original_symmetric_surrogate_preserved=True,no_refit_or_selection=True,primary_gate_passed=r['passed'],model=f['model'],
      series_source_sha256=hashlib.sha256(code.encode()).hexdigest(),wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
