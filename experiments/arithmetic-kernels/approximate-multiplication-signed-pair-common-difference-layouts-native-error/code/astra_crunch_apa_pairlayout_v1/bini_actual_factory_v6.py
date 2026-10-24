# Source parent SHA256: 16260481443ac88d5783d947d93a9905fe918276f8577a1e3ab1b79bfed487a6
# Pair packing is opt-in for all three orientations. No candidate/model integration.
# v6: 322/232 pack adjacent signed pairs into bb columns; dual223 uses aa rows.
"""Real public-fnp Bini factory. All three orientations; epsilon_power parameter.
transpose=False:322; True:transposed322=223; "232":cyclic232.
Caller supplies pubmax_B module with bound _SCRATCH.
Per-shape allocations/caches are not yet a production memory plan.
"""
def bini232_coefficients(e):
    U=[[0.]*10 for _ in range(6)];V=[[0.]*10 for _ in range(4)];W=[[0.]*10 for _ in range(6)]
    us=[{0:1},{0:1,3:1},{3:1},{5:1,4:e},{3:1,1:e},{0:1,1:e},{2:1,5:1},{2:1},{5:1},{2:1,4:e}]
    vs=[{3:1},{3:1,0:e},{2:1,3:1},{0:1,2:e},{2:1,0:-e},{3:1,1:e},{0:1,3:e},{0:1,1:1},{0:1},{1:1,3:-e}]
    ws=[{1:1/e,2:-1/e,4:1/e,0:-1/e},{5:1/e,0:-1/e},{4:1,3:-1,6:1},{1:1,5:-1,9:1},{3:1/e,8:-1/e},{6:1/e,7:-1/e,9:1/e,8:-1/e}]
    for j,co in enumerate(us):
        for i,v in co.items():U[i][j]=v
    for j,co in enumerate(vs):
        for i,v in co.items():V[i][j]=v
    for i,co in enumerate(ws):
        for j,v in co.items():W[i][j]=v
    U232rows=tuple(tuple(W[i][j]for i in (0,2,4,1,3,5))for j in range(10))
    V232rows=tuple(tuple(U[i][j]for i in range(6))for j in range(10))
    W232rows=tuple(tuple(V[i][j]for j in range(10))for i in (0,2,1,3))
    return U232rows,V232rows,W232rows

def make_bini(M,a,b,L,transpose=False,epsilon_power=-8,pair_cohorts=None,cohort_widths=None):
    import math
    if cohort_widths is not None:
        assert pair_cohorts is None
        cursor=0;pair_cohorts=[]
        for count in cohort_widths:
            count=int(count)
            assert count>0 and count%2==0
            pair_cohorts.append((cursor,count));cursor+=count
        assert cursor==int(b.shape[1])
        pair_cohorts=tuple(pair_cohorts)
    fn=M.fnp;arena=M._SCRATCH[0]
    def arr(sh):return fn.empty(sh,dtype=fn.float32)
    U232rows,V232rows,W232rows=bini232_coefficients(2.**epsilon_power)
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
    pair_inputs=[]
    if pair_cohorts is not None:
        assert transpose is True or transpose is False or transpose == '232'
        assert (cm if transpose is True else cn)%2==0
        cursor=0
        for start,count in pair_cohorts:
            start,count=int(start),int(count);half=count//2
            assert start==cursor and count>0 and count%2==0
            if transpose is True:
                assert start+count<=m
                pair_inputs.append((aa[start:start+count:2,:k],a[start:start+half,:]))
                pair_inputs.append((aa[start+1:start+count:2,:k],a[start+half:start+count,:]))
            else:
                assert start+count<=n
                pair_inputs.append((bb[:k,start:start+count:2],b[:,start:start+half]))
                pair_inputs.append((bb[:k,start+1:start+count:2],b[:,start+half:start+count]))
            cursor+=count
        assert cursor==(m if transpose is True else n)
    LA=arr((10,cm,ck));RB=arr((10,ck,cn));PC=arr((10,cm,cn))
    pcs=tuple(fn.unstack(PC,axis=0));parts=[(LA[i:i+P],RB[i:i+P],PC[i:i+P])for i in range(0,10,P)]
    packed=arr((ar*an,cm,cn))
    eps=fn.float32(2.**epsilon_power);inv=fn.float32(2.**(-epsilon_power))
    def run():
        # Only uncovered edge strips are zeroed, matching live Smirnov packing.
        if m<ar*cm:fn.copyto(aa[m:,:],0)
        if k<ak*ck:fn.copyto(aa[:m,k:],0)
        if k<ak*ck:fn.copyto(bb[k:,:],0)
        if n<an*cn:fn.copyto(bb[:k,n:],0)
        if pair_cohorts is None or transpose is not True:fn.copyto(destA,a)
        if pair_cohorts is None or transpose is True:fn.copyto(destB,b)
        if pair_cohorts is not None:
            for dest,source in pair_inputs:fn.copyto(dest,source)
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
            fn.stack(combine(uv[0],U232rows),out=LA)
            fn.stack(combine(uv[1],V232rows),out=RB)
            for i,(A,B,C) in enumerate(parts):ker.bb_batch(A,B,C,L,('bini',i))
            fn.stack(combine(pcs,W232rows),out=packed)
        else:
            a11,a12=av[0];a21,a22=av[1];a31,a32=av[2]
            b11,b12=bv[0];b21,b22=bv[1]
            # Boyer-Dumas poster formula, with shared input epsilon scales.
            ea12=fn.multiply(a12,eps);ea31=fn.multiply(a31,eps);eb11=fn.multiply(b11,eps);eb22=fn.multiply(b22,eps)
            S1=a11+a22;T1=b22+eb11;T2=b21+b22
            S3=a32+ea31;T3=b11+fn.multiply(b21,eps)
            S4=a22+ea12;T4=b21-eb11
            S5=a11+ea12;T5=b22+fn.multiply(b12,eps)
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
        if pair_cohorts is not None:
            # Owned fresh output, exactly as parent fn.copy. Undo pair order
            # during the already-required output materialization.
            answer=arr((n,m) if transpose is True else (m,n))
            for start,count in pair_cohorts:
                half=count//2
                if transpose is True:
                    fn.copyto(answer[:,start:start+half],z[start:start+count:2,:].T)
                    fn.copyto(answer[:,start+half:start+count],z[start+1:start+count:2,:].T)
                else:
                    fn.copyto(answer[:,start:start+half],z[:,start:start+count:2])
                    fn.copyto(answer[:,start+half:start+count],z[:,start+1:start+count:2])
            return answer
        return fn.copy(z.T) if transpose is True else fn.copy(z)
    evaluate=run
    def run():
        # Same invalidation/hold retirement as the production Smirnov caller.
        M._TR_LAST[0]=None
        try:return evaluate()
        finally:
            M._TR_LAST[0]=None
            M._BH_HOLD.clear()
            ker.cache.clear()
    return run,dict(L=L,P=P,orientation='232' if transpose=='232' else '223' if transpose else '322',child=[cm,ck,cn],arena_elements=7**L*P*(r*q+q*c+r*c),outer_elements=10*(cm*ck+ck*cn+cm*cn),padding_elements=ar*ak*cm*ck+ak*an*ck*cn)
