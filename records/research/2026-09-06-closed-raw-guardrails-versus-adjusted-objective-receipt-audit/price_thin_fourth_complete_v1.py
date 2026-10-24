"""Complete conservative shape-only q4/s4 cost screen, no target access."""
from collections import defaultdict
from itertools import combinations,product
import json
from pathlib import Path
import numpy as np
from flopscope.accounting import qr_cost,eigh_cost,solve_cost
from thin_fourth_shape_cost_v1 import Ledger,Shape,Backend
import fourth_trace_full_queries_v1 as full
import fourth_trace_fused_cp_queries_v1 as fusedcp
import fourth_trace_third_queries_v1 as third
import fourth_low_rank_source_v1 as source

HERE = Path(__file__).resolve().parent
OUT = HERE/'thin_fourth_complete_cost_v1.json'
assert not OUT.exists()
TERMS = json.loads((HERE/'thin_fourth_primary_terms_v1.json').read_text())['terms']
COEF = json.loads((HERE/'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
BANK = json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']


def bank_mm(m,k,n):
    if not m or not k or not n:return 0.
    if n>8192:return sum(bank_mm(m,k,min(8192,n-j)) for j in range(0,n,8192))+m*n
    if k>8192:return sum(bank_mm(m,min(8192,k-j),n) for j in range(0,k,8192))+(len(range(0,k,8192))-1)*m*n
    plain=2*m*k*n-m*n
    if any(t%32 for t in (m,k,n)):return float(plain)
    depths=(3,4,5) if max(m,k,n)<=1024 else (4,)
    return min(plain,*(sum(a*b for a,b in zip(COEF[str(d)],(m*k*n,m*k,k*n,m*n))) for d in depths))


def repeated_correction(L,n):
    """Same unfused non-third term operations as primary small certificate.

    All degree<=4 slices are available from the parent/new preplane reads.
    Matrix core/metric used by block expansion are constructed and charged
    separately. Only symbolic terms, not weight or target tensors, loaded.
    """
    xp=Backend(L); C,A,B,K21,K22=[L.shape(n,n) for _ in range(5)]
    v=L.shape(n); ans=xp.zeros((n,n)); h=[v]*7; w=[v]*7
    def small(vec):
        return v if len(vec)==1 else (C if sum(vec)==2 else K21 if sum(vec)==3 else K22)
    def expansion(block):
        axes=tuple(i for i,x in enumerate(block) if x)
        vec=tuple(x for x in block if x)
        if len(axes)<=2:return [(1.,[(axes,small(vec))])]
        assert sum(block)==4
        legs=tuple(i for i,x in enumerate(block) for _ in range(x)); out=[]
        for chosen in combinations(range(4),2):
            other=tuple(i for i in range(4) if i not in chosen); factors=[]
            for matrix,pair in ((A,chosen),(B,other)):
                i,j=legs[pair[0]],legs[pair[1]]
                factors.append(((i,),matrix.diag()) if i==j else ((i,j),matrix))
            out.append((1/6,factors))
        return out
    for term in TERMS:
        ks=term['ks']; initial=[h[ks[0]],w[ks[1]],w[ks[2]]]
        for summands in product(*(expansion(b) for b in term['blocks'])):
            coef=term['coefficient']; nodes=[x.copy() for x in initial]; edges={}
            for c,factors in summands:
                coef*=c
                for axes,arr in factors:
                    if len(axes)==1:nodes[axes[0]]*=arr
                    else:
                        i,j=axes
                        if i>j:i,j,arr=j,i,arr.T
                        edges[i,j]=edges.get((i,j),1.)*arr
            ones=xp.ones((n,n))
            e01,e02,e12=(edges.get(pair,ones) for pair in ((0,1),(0,2),(1,2)))
            v0,v1,v2=nodes
            ans+=coef*((v0*v1)[:,None]*v2[None,:]*e01.diag()[:,None]*e02*e12*ones
                       +v1[:,None]*(v0*v2)[None,:]*e01.T*e02.diag()[None,:]*e12*ones)
    return ans


def pair31_cost(L,n):
    # Additional Gaussian6/Edgeworth2 covariance(ReLU^3,ReLU), with all
    # existing standardization/coefficient planes reused. Power3 derivative
    # vectors cost separately: generous 1000 f64 operations per neuron.
    L.add('power3_vector_allowance',2*1000*n)
    xp=Backend(L); v=L.shape(n); C=L.shape(n,n)
    F=G=[v]*7; t21=t22=t31=rho=C; t3=t4=v
    ans=xp.zeros_like(C); rr=xp.ones_like(C)
    for r in range(7):
        if r:ans=ans+rr*xp.outer(F[r],G[r])
        if r<=2:
            ec=(t21*xp.outer(F[r+2],G[r+1])+t21.T*xp.outer(F[r+1],G[r+2]))/2
            ec+=t22*xp.outer(F[r+2],G[r+2])/4
            ec+=(t31*xp.outer(F[r+3],G[r+1])+t31.T*xp.outer(F[r+1],G[r+3]))/6
            if r:
                ec+=(xp.outer(t3*F[r+3],G[r])+xp.outer(F[r],t3*G[r+3]))/6
                ec+=(xp.outer(t4*F[r+4],G[r])+xp.outer(F[r],t4*G[r+4]))/24
            ans+=rr*ec
        rr=rr*rho/(r+1)
    p31=ans*xp.outer(v*v*v,v)
    # Convert covariance powers to central cumulant31 (diag reuses marginal4).
    return p31-3*v[:,None]*C+3*(v*v-v)[:,None]*C


def mode_cost(L,n,q):
    C,U=L.shape(n,n),L.shape(n,q)
    L.add('qr',2*qr_cost(n,q))
    for _ in range(3):
        U=C@U
        L.add('qr',2*qr_cost(n,q))
    CU=C@U; H=U.T@CU
    L.add('small_eigh',2*eigh_cost(q))
    CU@L.shape(q,q)
    L.add('mode_scalar_allowance',2*(20*n*q+50*q*q))


def core_cost(L,n,q,degree=32):
    C,U,RU=L.shape(n,n),L.shape(n,q),L.shape(n,q)
    H=U.T@C@U;L.add('small_eigh',2*eigh_cost(q))
    U@L.shape(q,q);RU@L.shape(q,q)
    CU=C@U;F=CU-U*L.shape(q)[None,:]
    Rhigh=U.T@RU;Rhigh=(Rhigh+Rhigh.T)/2
    Rlow=RU-U@Rhigh
    (C*C).sum();(CU*CU).sum()
    b=q*(q+1);X=L.shape(n,b)
    def Q(X):return X-U@(U.T@X)
    def D(X):return Q(C@Q(X))
    previous=X; current=D(X)/1.
    inverse=X/L.shape(b)[None,:]+current*L.shape(b)[None,:]
    for _ in range(2,degree+1):
        following=2*D(current)/1.-previous
        inverse+=following*L.shape(b)[None,:]
        previous,current=current,following
    Z=[L.shape(n,q) for _ in range(q)];Y=L.shape(n,q)
    h=L.shape(q);H=L.shape(q,q)
    R0=(1.*H+2*(H@H+F.T@F))/3
    # Literal small affine system construction: offset, each symmetric
    # basis column, alpha column, then final low reconstruction.
    for _ in range(1+q*(q+1)//2+1):
        S=L.shape(q,q)
        lows=[]
        for j in range(q):
            lows.append(Y[:,j]-1.*Z[j][:,j]-2*Z[j]@S[:,j]-2*1.*F[:,j])
        low=Backend(L).stack(lows,axis=1)
        high=(1.*S+1.*H+2*(H@S+S@H+F.T@low+low.T@F))/6+1.*R0
        (H*S).sum();(F*low).sum()
    k=q*(q+1)//2+1
    L.add('small_solve',2*solve_cost(k,1))
    for j in range(q):Y[:,j]-1.*Z[j][:,j]-2*Z[j]@L.shape(q)-2*F[:,j]
    Y+.5*U@L.shape(q,q)
    # Mode conditioning diagnostics, scalar recurrence powers, system
    # assembly, trace computation and normalizations: conservative allowance.
    L.add('core_scalar_allowance',2*(100*n*q+1000*q*q*degree))


def per_layer(n,q,s,R,prev_a,layer,mat_rate):
    L=Ledger(mat_rate);xp=Backend(L)
    full.np=third.np=fusedcp.np=source.np=xp
    v=L.shape(n);C=L.shape(n,n);X=L.shape(n,q);V=L.shape(n,s)
    # Conservative: even the first Gaussian-only layer is charged full
    # geometry, generic third and core kernels. Final scalar layer needs
    # only pushed anisotropic core and scalar4 read.
    if layer:
        L.group='core_linear_push_and_preplanes'
        W=L.shape(prev_a,n);Z=L.shape(prev_a,2*q)
        Zp=W.T@Z
        A=(Zp*L.shape(2*q))@Zp.T
        (A*C).sum(1)
        L.add('general_preplane_allowance',2*40*n*n)
    if layer==15:
        return L
    L.group='mode_geometry_twice'
    mode_cost(L,n,q);mode_cost(L,n,s)
    L.group='post31_pair'
    pair31_cost(L,n)
    L.group='non_third_repeated_correction'
    repeated=repeated_correction(L,n)
    L.group='complete_trace_queries'
    otherM=third.ThirdContraction(L.shape(n,0),L.shape(0),L.shape(prev_a,prev_a),L.shape(prev_a,n),v,C)
    other=third.DistinctQueries(otherM,C,V)
    cq=fusedcp.FusedCPQueries(L.shape(n,R),L.shape(R),C,V,other)
    f=full.fused_query(C,C,C,C,C,[v]*7,[v]*7,V,cq,repeated,X)
    full.complete_trace_query(f,C,C,C,X)
    L.group='cross_core_projection'
    core_cost(L,n,q)
    L.group='core_factorization'
    L.add('thin_qr',2*qr_cost(n,2*q));L.add('small_eigh',2*eigh_cost(2*q))
    L.shape(n,2*q)@L.shape(2*q,2*q)
    L.add('core_factor_scalar_allowance',2*50*n*q)
    if layer:
        L.group='new_source_creation_and_repair'
        p,lam=source.source_factors(L.shape(n,2*q),L.shape(2*q),V,v,v,v)
        # Source construction uses exactly154 columns but bank kernels must
        # pad160. Price the current repeated-plane calculation on padded160.
        count=((p.shape[1]+31)//32)*32
        pp=L.shape(n,count)
        (pp*pp)@(pp*L.shape(count)).T
        # Exact repeated target from Q31,Q22, source T_iij and coordinate
        # star compensation: conservative scalar allowance 50 per pair.
        L.add('source_repeated_and_star_allowance',2*50*n*n)
    return L


rows=[]
for rank in (35,100):
    folder=HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
    x=np.load(folder/'x.npy');active=np.load(folder/'alive.npy').sum(1).astype(int)
    births=((np.count_nonzero(np.minimum(x[:,:,5],1-x[:,:,5])>=.01,axis=1)+31)//32*32).clip(max=1024)
    bank=next(r['flops'] for r in BANK if r['rank']==rank)
    for name,rate in (('all_f64_new_kernels',2),('f32_gemm_f64_elementwise_UNCERTIFIED',1)):
        R=extra=0;layers=[];groups=defaultdict(float)
        for l in range(16):
            a=int(active[l]);prev=1024 if l==0 else int(active[l-1])
            L=per_layer(1024,4,4,R+extra,prev,l,rate)
            L.group='future_new_source_carry'
            L.add('f32_bank_push',bank_mm(1024,prev,R+extra)-bank_mm(1024,prev,R))
            if l<15:
                L.add('f32_bank_plane',bank_mm(a,R+extra,a)-bank_mm(a,R,a))
                L.add('f32_bank_product_and_mask',4*a*extra+2*1024*extra)
            L.add('f64_diagonal_cubic',8*1024*extra)
            for k,value in L.counts.items():groups[k.split('/')[0]]+=value
            layers.append(dict(layer=l,active=a,previous_active=prev,base_columns=R,extra_columns=extra,
                               total_extra=L.total,costs=dict(L.counts)))
            R+=2*int(births[l])
            if 0<l<15:extra+=160
        total=sum(z['total_extra'] for z in layers)
        rows.append(dict(rank=rank,policy=name,new_gemm_dtype_rate=rate,
                         current_first_context_measured_flops=bank,projected_extra_flops=total,
                         projected_flop_multiplier=1+total/bank,projected_utilization=(bank+total)/2**41,
                         cost_gate_pass=1+total/bank<=1.40,
                         raw_needed_for_4e9=4e-9/((bank+total)/2**41),
                         raw_needed_for_1e9=1e-9/((bank+total)/2**41),groups=dict(groups),layers=layers))
receipt=dict(rows=rows,cost_gate_multiplier=1.40,
             qualification='Conservative analytical shape-only architecture estimate, not actual flopscope metering, runtime, memory peak or score. New kernels use full1024 geometry; primary unfused31-term repeated repair is included. Public linear-algebra costs and measured bank f32 kernel polynomials with frozen shape tapes; new mixed GEMMs require a separate precision certificate. No target access. Scalar allowances and duplicate first-layer/geometry work are explicit; interpreter/copy/allocator costs are not FLOPs and remain release gates.')
OUT.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({**receipt,'rows':[{k:v for k,v in r.items() if k!='layers'} for r in rows]},indent=2))
