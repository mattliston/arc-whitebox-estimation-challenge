"""Official small-array padding selection and exact BD/born consistency checks."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,ast,hashlib,json,resource,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--official',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
sys.path.insert(0,str(a.official))
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_WEIGHTS_FILE')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads(a.manifest.read_text());rec=m['candidates'][0];parent=Path(rec['science']['parent_path']);selector=Path(__file__).with_name('selector_v1.py')
assert sha(parent)==m['parent_source_sha256'] and sha(selector)==m['selector_sha256']
source=parent.read_text();tree=ast.parse(source);defs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
tail=Path(rec['append_tail']).read_text();newdefs={n.name:n for n in ast.parse(tail).body if isinstance(n,ast.FunctionDef)}
def env(backend):
    d=dict(fnp=backend,_E2_LAYER={})
    exec(compile(ast.Module(body=[defs['_E2Hold'],defs['select']],type_ignores=[]),'<original selector>','exec'),d)
    exec(compile(selector.read_text(),str(selector),'exec'),d)
    return d
en,ef=env(np),env(fnp)
def put(x):
    with fs.BudgetContext(flop_budget=2**50,quiet=True):return fnp.asarray(x)
def measure(call):
    with fs.BudgetContext(flop_budget=2**50,quiet=True) as b:y=call()
    return y,dict(flops=b.flops_used,calls=len(b._op_log))
rng=np.random.default_rng(202610210238);records=[]
for n,required in ((32,0),(32,5),(64,32),(64,1),(65,35),(96,35),(96,64)):
 for strided in (False,True):
    score=np.concatenate((np.linspace(.011,.49,required),np.linspace(0,.0099,n-required))).astype('d');rng.shuffle(score)
    base=rng.normal(size=(n,n)).astype('f');np.fill_diagonal(base,0)
    av=np.exp(rng.uniform(-2,2,n));tv=rng.normal(size=n);sv=tv-rng.uniform(.1,3,n)
    if strided:
        backing=np.zeros((2*n,2*n),dtype='f');backing[::2,::2]=base;basef=put(backing)[::2,::2]
    else:basef=put(base)
    sf,af,tf,vf=map(put,(score,av,tv,sv));old,_=measure(lambda:ef['select'](sf,.01));prepared,_=measure(lambda:ef['_o21_birth_prepare'](sf,.01))
    assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(old,prepared))
    before=tuple(np.asarray(x).tobytes() for x in (basef,sf,af,tf,vf,*prepared))
    newmask,price=measure(lambda:ef['_o21_birth_refine'](basef,af,tf,vf,prepared[1]));out=ef['_o21_birth_finish'](*prepared)
    assert out[1] is newmask and not ef['_E2_LAYER']
    eni,enm=en['_o21_birth_prepare'](score,.01);en['_o21_birth_refine'](base,av,tv,sv,enm);oracle=en['_o21_birth_finish'](eni,enm)
    actual,mask=map(np.asarray,out);assert np.array_equal(actual,oracle[0]) and np.array_equal(mask,oracle[1])
    assert len(actual)==len(np.asarray(old[0])) and len(set(actual.tolist()))==len(actual)
    assert np.all(mask[score>=.01]) and np.array_equal(mask,np.isin(np.arange(n),actual))
    gate_order=np.argsort(-score,kind='stable');assert np.array_equal(actual,gate_order[mask[gate_order]])
    assert before==tuple(np.asarray(x).tobytes() for x in (basef,sf,af,tf,vf,*prepared))
    padding=len(actual)-required
    if padding==0 or len(actual)==n:assert out[0] is prepared[0] and out[1] is prepared[1] and price['flops']==0
    # Equal-energy ties exactly recover the original selection, with no value cache.
    zeros=put(np.zeros_like(base));p2,_=measure(lambda:ef['_o21_birth_prepare'](sf,.01));measure(lambda:ef['_o21_birth_refine'](zeros,af,tf,vf,p2[1]));z=ef['_o21_birth_finish'](*p2)
    assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(old,z))
    assert ef['_o21_birth_refine'](basef,af,tf,vf,prepared[1]) is prepared[1]
    records.append(dict(n=n,required=required,padding=padding,strided=strided,price=price,same_count=True,above_floor_preserved=True,mask_roster_consistent=True,stable_gate_order=True,input_immutable=True,zero_energy_original_bytes=True,no_stale_context=True))

# Explicit all-distinct tensor energy oracle. This verifies the score's formula,
# separate from intentionally rounded f32 products used to rank actual bases.
n=7;base=rng.normal(size=(n,n));np.fill_diagonal(base,0);av=rng.normal(size=n);tv=rng.normal(size=n);sv=rng.normal(size=n)
score=en['_o21_birth_distinct_scores'](base,av,tv,sv)
distinct=np.fromfunction(lambda i,j,k:(i!=j)&(i!=k)&(j!=k),(n,n,n),dtype=int)
for j in range(n):
    p=base[:,j].copy();q=p.copy();p[j]+=tv[j];q[j]+=sv[j]
    T=av[j]*(np.einsum('i,j,k->ijk',p,p,p)-np.einsum('i,j,k->ijk',q,q,q))
    assert np.isclose(np.sum(T[distinct]**2),3*score[j],rtol=1e-12,atol=1e-12)

# Execute the actual old/new BD function bodies with identical small public
# helper fixtures. The candidate delta must equal the original delta computed
# with the new mask, and moving the independent born work must preserve bytes.
bdrecords=[]
def bd_env(new):
    e=env(fnp);e.update(_BD_J1_PHYS=(1.,1.,1.),_S19_BD_ADD=(.03,-.02,.01),_BD32_PARENT=lambda *args:('fallback',args[6]))
    states={}
    def state(n):
        if n not in states:
            D=fnp.empty((3,n,n),dtype=fnp.float32);states[n]=dict(D=D,Dv=fnp.unstack(D),D12=D[1:],diag=fnp.asarray([b*n*n+i*(n+1) for b in range(3) for i in range(n)]),old32=fnp.empty((n,n),dtype=fnp.float32))
        return states[n]
    e.update(_bd32_state=state,_ps_k2132=lambda x:x.astype(fnp.float32),_ps_pre32=lambda x:x.astype(fnp.float32),_idc_get=lambda name,obj,axis,f:f(),_g5_j1=lambda a,b:(fnp.asarray(a),fnp.zeros(3),fnp.zeros((3,1))),_g5_arr=lambda x:fnp.asarray(x),_g5_mulrow=lambda x,y:x*y[:,None],_o18_bd_literal_column=lambda x:fnp.asarray(x)[:,None],_g5_chain=lambda x,y,z:(x*y)*z)
    exec(compile(ast.Module(body=[newdefs['_bd_update'] if new else defs['_bd_update']],type_ignores=[]),'<actual BD>','exec'),e)
    return e
for n,required in ((32,5),(64,32),(65,35)):
 for matrix_coeff in (False,True):
    score=np.linspace(0,.009,n);score[:required]=np.linspace(.02,.4,required)
    arrays=[rng.normal(size=(n,n)).astype('f') for _ in range(3)];np.fill_diagonal(arrays[0],0)
    bp,cp,kp=map(put,arrays);alive=put(rng.uniform(size=n)>.15);av,tv,sv=[put(rng.normal(size=n))for _ in range(3)];sf=put(score)
    aux={k:put(rng.uniform(.2,1.2,n))for k in('alpha','sigma','gate')};d=put(rng.uniform(.2,1.2,n));coefs=put(rng.normal(scale=.02,size=(3,n) if matrix_coeff else (3,)))
    enew,eold=bd_env(True),bd_env(False)
    prepared,_=measure(lambda:enew['_o21_birth_prepare'](sf,.01));base_delta,_=measure(lambda:enew['_bd_update'](bp,cp,kp,aux,d,alive,prepared[1],av,tv,sv,coefs));chosen=enew['_o21_birth_finish'](*prepared)
    control,_=measure(lambda:eold['_bd_update'](bp,cp,kp,aux,d,alive,chosen[1],av,tv,sv,coefs))
    assert all(np.asarray(x).tobytes()==np.asarray(y).tobytes() for x,y in zip(base_delta,control))
    birth=put(rng.uniform(.1,2,n));beta=put(rng.normal(size=n));threeu2=put(rng.uniform(0,2,n));delta=put(rng.uniform(.1,2,n))
    def born(mask,plane):
        bb=birth*mask;B=fnp.einsum('ij,ij,j->ij',bp,bp,bb.astype(fnp.float32),optimize=False)+fnp.einsum('i,ji->ij',(beta*mask).astype(fnp.float32),bp,optimize=False)
        diag=bb*(threeu2+delta*delta)/4;B=fnp.add(B,plane,out=B);diag=diag+fnp.diag(plane);fnp.fill_diagonal(B,diag);return B
    bnew,_=measure(lambda:born(chosen[1],base_delta[1]));bold,_=measure(lambda:born(chosen[1],control[1]));assert np.asarray(bnew).tobytes()==np.asarray(bold).tobytes()
    bdrecords.append(dict(n=n,required=required,matrix_coeff=matrix_coeff,actual_BD_body_new_mask_oracle_bytes=True,born_plane_same_mask_bytes=True))
receipt=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),successful=True,official_version=fs.__version__,manifest_sha256=sha(a.manifest),selector_sha256=sha(selector),script_sha256=sha(__file__),selector_cases=records,BD_body_cases=bdrecords,tensor_energy_oracle_cases=7,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,no_estimator_import=True,no_prediction=True,no_truth=True,accuracy_gain_unknown=True)
assert receipt['peak_rss_bytes']<1_000_000_000
with a.out.open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(dict(successful=True,selector_cases=len(records),BD_cases=len(bdrecords),peak_rss_bytes=receipt['peak_rss_bytes'])))
