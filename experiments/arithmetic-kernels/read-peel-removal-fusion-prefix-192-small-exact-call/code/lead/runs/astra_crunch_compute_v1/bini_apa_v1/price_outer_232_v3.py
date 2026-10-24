"""One Bini APA outer layer over actual current alternative-basis tree kernels.
Real flopscope 0.12.2 charging, metadata backend only; no model or numeric kernel.
Explicit padding, input packing, 10 products, all linear ops and output packing.
"""
import collections, hashlib, importlib.util, json, math, resource
from pathlib import Path
D=Path(__file__).resolve().parent; O=D.parents[3]
s=importlib.util.spec_from_file_location('probe',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py')
p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
p.SOURCE=O/'lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py'
p.SOURCE_SHA=hashlib.sha256(p.SOURCE.read_bytes()).hexdigest()
orig=p.backend
def backend(fn,*args,**kwargs):
    name=getattr(fn,'__name__','')
    if name in ('copy','astype') and hasattr(fn,'__self__'):
        p.CALLS[name]+=1;a=fn.__self__
        dt=(args[0] if args else kwargs.get('dtype',a.dtype)) if name=='astype' else a.dtype
        return p.virtual(a.shape,dt).view(type(a))
    return orig(fn,*args,**kwargs)
p.backend=backend
M,_=p.load();fn=p.fnp;fs=p.flops
M._WINOGRAD=True;M._AB_ON=True
with fs.BudgetContext(flop_budget=2**60,quiet=True):
    arena=fn.empty((M._PACKED_ELEMENTS,),dtype=fn.float32)
    M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None]
    M._bind_push6(arena)
from flopscope.numpy import ndarray
views=collections.Counter();old_get=ndarray.__getitem__;old_T=ndarray.T
def get(self,key):views['getitem']+=1;return old_get(self,key)
def getT(self):views['T']+=1;return old_T.__get__(self,type(self))
ndarray.__getitem__=get;ndarray.T=property(getT)
def arr(sh):return fn.empty(sh,dtype=fn.float32)
def measure(f):
    M._BH_HOLD.clear();M._TR_LAST[0]=None;p.CALLS.clear();views.clear()
    with fs.BudgetContext(flop_budget=2**60,quiet=True) as c:z=f()
    return dict(flops=int(c.flops_used),requests=sum(p.CALLS.values())+sum(views.values()),charged=len(c._op_log),backend=dict(p.CALLS),views=dict(views))

def make_bini(a,b,L,transpose=False):
    # Transposition swaps m,n and gives the cyclic dual (2,2,3).
    if transpose is True:a,b=b.T,a.T
    ar,ak,an=(2,3,2) if transpose=='232' else (3,2,2)
    m,k,n=int(a.shape[0]),int(a.shape[1]),int(b.shape[1]);u=1<<L
    cm,ck,cn=math.ceil(m/(ar*u))*u,math.ceil(k/(ak*u))*u,math.ceil(n/(an*u))*u
    r,q,c=cm//u,ck//u,cn//u
    P=next((x for x in (10,5,2,1) if 7**L*x*(r*q+q*c+r*c)<=391602176),None)
    if P is None:return None
    fam=('bini',L,P,r,q,c)
    # Scratch is the real existing arena. No larger symbolic capacity permitted.
    env=M._SmxEnvironment();env.fnp=fn;env._IPT_REGION=M._IPT_REGION;env._BH_HOLD=M._BH_HOLD;env._TR_LAST=M._TR_LAST
    env._SCRATCH=M._SCRATCH;env._rs_nat_perm=M._rs_nat_perm;env._rs_basis_views=M._rs_basis_views
    env._sm40_cache={};env._sm40_io_cache={};env._sm40_frames={'bini':M._astra_sm40_bind_family(fn,env,arena,fam)}
    ker=M._SmxKernel(env,{'name':'bini','prepared':{}});ker.family=fam
    aa=arr((ar*cm,ak*ck));bb=arr((ak*ck,an*cn))
    av=[[aa[i*cm:(i+1)*cm,j*ck:(j+1)*ck]for j in range(ak)]for i in range(ar)]
    bv=[[bb[i*ck:(i+1)*ck,j*cn:(j+1)*cn]for j in range(an)]for i in range(ak)]
    destA=aa[:m,:k];destB=bb[:k,:n]
    LA=arr((10,cm,ck));RB=arr((10,ck,cn));PC=arr((10,cm,cn))
    pcs=tuple(fn.unstack(PC,axis=0));parts=[(LA[i:i+P],RB[i:i+P],PC[i:i+P])for i in range(0,10,P)]
    packed=arr((ar*an,cm,cn))
    eps=fn.float32(2**-10);inv=fn.float32(2**10)
    def run():
        # Only uncovered edge strips are zeroed, matching live Smirnov packing.
        if m<ar*cm:fn.copyto(aa[m:,:],0)
        if k<ak*ck:fn.copyto(aa[:m,k:],0)
        if k<ak*ck:fn.copyto(bb[k:,:],0)
        if n<an*cn:fn.copyto(bb[:k,n:],0)
        fn.copyto(destA,a);fn.copyto(destB,b)
        if transpose=='232':
            uv=[[x for row in av for x in row],[x for row in bv for x in row]]
            def combine(inputs, rows):
                cache={};result=[]
                for row in rows:
                    coeffs=[(i,v) for i,v in enumerate(row) if v]
                    ab={abs(v)for _,v in coeffs};factor=next(iter(ab)) if len(ab)==1 else 1.
                    pos=[(i,v/factor)for i,v in coeffs if v>0];neg=[(i,-v/factor)for i,v in coeffs if v<0]
                    def term(i,v):
                        if v==1:return inputs[i]
                        key=(i,v)
                        if key not in cache:cache[key]=fn.multiply(inputs[i],fn.float32(v))
                        return cache[key]
                    if pos:
                        z=term(*pos[0])
                        for iv in pos[1:]:z=fn.add(z,term(*iv))
                        for iv in neg:z=fn.subtract(z,term(*iv))
                    else:
                        z=fn.multiply(term(*neg[0]),fn.float32(-1))
                        for iv in neg[1:]:z=fn.subtract(z,term(*iv))
                    if factor!=1:z=fn.multiply(z,fn.float32(factor))
                    result.append(z)
                return result
            fn.stack(combine(uv[0],U232.T),out=LA)
            fn.stack(combine(uv[1],V232.T),out=RB)
            for i,(A,B,C) in enumerate(parts):ker.bb_batch(A,B,C,L,('bini',i))
            fn.stack(combine(pcs,W232),out=packed)
        else:
            a11,a12=av[0];a21,a22=av[1];a31,a32=av[2]
            b11,b12=bv[0];b21,b22=bv[1]
            # Boyer-Dumas poster formula, with shared input epsilon scales.
            ea12=eps*a12;ea31=eps*a31;eb11=eps*b11;eb22=eps*b22
            S1=a11+a22;T1=b22+eb11;T2=b21+b22
            S3=a32+ea31;T3=b11+eps*b21
            S4=a22+ea12;T4=b21-eb11
            S5=a11+ea12;T5=b22+eps*b12
            S6=a21+a32;T6=b11+eb22;T7=b11+b12
            S9=a21+ea31;T9=b12-eb22
            fn.stack((a11,S1,a22,S3,S4,S5,S6,a21,a32,S9),out=LA)
            fn.stack((b22,T1,T2,T3,T4,T5,T6,T7,b11,T9),out=RB)
            for i,(A,B,C) in enumerate(parts):ker.bb_batch(A,B,C,L,('bini',i))
            p0,p1,p2,p3,p4,p5,p6,p7,p8,p9=pcs
            c11=((p1-p2)+p4-p0)*inv;c12=(p5-p0)*inv
            c21=p4-p3+p6;c22=p1-p5+p9
            c31=(p3-p8)*inv;c32=((p6-p7)+p9-p8)*inv
            fn.stack((c11,c12,c21,c22,c31,c32),out=packed)
        out=fn.reshape(fn.transpose(fn.reshape(packed,(ar,an,cm,cn)),(0,2,1,3)),(ar*cm,an*cn))
        z=out[:m,:n]
        return fn.copy(z.T) if transpose is True else fn.copy(z)
    return run,dict(L=L,P=P,orientation='232' if transpose=='232' else '223' if transpose else '322',child=[cm,ck,cn],arena_elements=7**L*P*(r*q+q*c+r*c),outer_elements=10*(cm*ck+ck*cn+cm*cn),padding_elements=ar*ak*cm*ck+ak*an*ck*cn)

# Tensor cyclic dual: (3,2,2) -> (2,2,3) -> (2,3,2).
import numpy as np
e=2.**-10
U=np.zeros((6,10));V=np.zeros((4,10));W=np.zeros((6,10))
us=[{0:1},{0:1,3:1},{3:1},{5:1,4:e},{3:1,1:e},{0:1,1:e},{2:1,5:1},{2:1},{5:1},{2:1,4:e}]
vs=[{3:1},{3:1,0:e},{2:1,3:1},{0:1,2:e},{2:1,0:-e},{3:1,1:e},{0:1,3:e},{0:1,1:1},{0:1},{1:1,3:-e}]
ws=[{1:1/e,2:-1/e,4:1/e,0:-1/e},{5:1/e,0:-1/e},{4:1,3:-1,6:1},{1:1,5:-1,9:1},{3:1/e,8:-1/e},{6:1/e,7:-1/e,9:1/e,8:-1/e}]
for j,co in enumerate(us):
    for i,v in co.items():U[i,j]=v
for j,co in enumerate(vs):
    for i,v in co.items():V[i,j]=v
for i,co in enumerate(ws):
    for j,v in co.items():W[i,j]=v
U232=W.reshape(3,2,10).transpose(1,0,2).reshape(6,10)
V232=U
W232=V.reshape(2,2,10).transpose(1,0,2).reshape(4,10)
rg=np.random.default_rng(20261016)
for _ in range(20):
    a=rg.normal(size=(2,3));b=rg.normal(size=(3,2))
    got=(W232@((U232.T@a.ravel())*(V232.T@b.ravel()))).reshape(2,2)
    assert np.linalg.norm(got-a@b)<.05*np.linalg.norm(a)*np.linalg.norm(b)
coverage=[];shapes=collections.Counter()
for path in sorted((O/'lead/runs/opus_small_v1/shapes').glob('shapes_*.json')):
    d=json.loads(path.read_text());hits=[]
    for row in d['rows']:
        if row['entry']=='_mm' and row['site']=='_astra_bcc_newborn_impl' and row['b'][1]>1024:
            sh=(*row['a'],row['b'][1]);shapes[sh]+=1;hits.append(list(sh))
    coverage.append(dict(seed=d['seed'],total=d['total'],hits=hits))
rows=[]
for shape,count in sorted(shapes.items()):
    m,k,n=shape
    with fs.BudgetContext(flop_budget=2**60,quiet=True):a=arr((m,k));b=arr((k,n))
    measure(lambda:M._mm(a,b));base=measure(lambda:M._mm(a,b))
    measure(lambda:M._pushsm_mm(a,b));sm=measure(lambda:M._pushsm_mm(a,b))
    trials=[]
    for trans in ('232',):
        for L in (4,5,6):
            with fs.BudgetContext(flop_budget=2**60,quiet=True):spec=make_bini(a,b,L,trans)
            if spec is None:continue
            call,desc=spec;measure(call);v=measure(call)
            trials.append(dict(**desc,**v,saving_vs_base=base['flops']-v['flops'],saving_vs_pushsm=sm['flops']-v['flops']))
    best=min(trials,key=lambda r:r['flops'])
    row=dict(shape=shape,count=count,base=base,pushsm=sm,best=best,trials=trials);rows.append(row)
    print(json.dumps(dict(shape=shape,count=count,base=base['flops'],pushsm=sm['flops'],best={k:best[k]for k in ['orientation','L','P','flops','saving_vs_base','saving_vs_pushsm']})),flush=True)
byshape={tuple(r['shape']):r for r in rows}
for c in coverage:
    c['saving_vs_base']=sum(max(0,byshape[tuple(sh)]['best']['saving_vs_base'])for sh in c['hits'])
    c['saving_vs_pushsm']=sum(max(0,byshape[tuple(sh)]['best']['saving_vs_pushsm'])for sh in c['hits'])
    c['saving_base_pct']=100*c['saving_vs_base']/c['total'];c['saving_pushsm_pct']=100*c['saving_vs_pushsm']/c['total']
out=dict(status='METADATA_ONLY_REAL_TARIFF_NOT_NUMERICALLY_QUALIFIED',source=str(p.SOURCE),source_sha=p.SOURCE_SHA,rows=rows,coverage=coverage,payload_bytes=p.PAYLOAD[0],max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations=['Warm recurring arithmetic only; no prebind/init bill included.','Counter requests are wrapper operations, not residual instructions.','Current _mm and exact pushsm2 comparator from same source; source includes other exact overlays.','Uses existing capped arena and actual Smx bb_batch arithmetic, all outer ops explicit.','Output materialization is charged per call; returned output is contiguous.'])
(D/'OUTER_232_TARIFF_v3.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([{k:v for k,v in c.items()if k!='hits'}for c in coverage],indent=2))
