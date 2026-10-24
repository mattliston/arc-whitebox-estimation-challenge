"""Fixed independent dual/VJP/checkpoint diagnostic; no imports of numeric APIs."""
import hashlib
import weakref

SEED=202609131341
ABS_LIMIT=1e-8
REL_LIMIT=1e-5
SOURCE_UPPER=60_000_000_000
PRODUCTION_SHAPES=((1984,896,384),(896,1984,384),(384,896,384),
                   (384,384,384),(896,384,384))
DENSE_FORWARD_F=sum(2*m*k*n for m,k,n in PRODUCTION_SHAPES)
DERIVED_BASE_F=4*4_162_560_000+8*DENSE_FORWARD_F+3_000_000_000+1_000_000_000
if not (DENSE_FORWARD_F==3_372_220_416 and
        DERIVED_BASE_F==47_628_003_328 and DERIVED_BASE_F<SOURCE_UPPER<100_000_000_000):
    raise RuntimeError('Fixed complete source bound must hold before numerical work')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def evaluate(bridge,prepared,bundle,np,torch,workspace,refs,remember,tick):
    from torch.utils.checkpoint import checkpoint
    # All directions/cotangents fixed before observations, no FD or tuning.
    rng=np.random.Generator(np.random.PCG64DXSM(SEED))
    def host(x):return x.detach().numpy().astype(np.float64)
    def norm(x):return float(np.sqrt(np.sum(x*x,dtype=np.float64)))
    def raw(x):return x.detach().numpy().tobytes()
    def tensor(x,dtype,trainable=False):
        answer=torch.tensor(x,dtype=dtype,device='cpu',requires_grad=trainable)
        refs.append(weakref.ref(answer));return answer
    def random(shape,scale=1.):
        return rng.standard_normal(shape)*scale
    def direction(shape,dtype):
        d=random(shape);d=d/norm(d)
        return d.astype(np.float32).astype(np.float64) if dtype==torch.float32 else d
    def primals(run,x):
        J,c,O=x[:3];w=c*c;B=run.mm(J.T,O);H=w[:,None]*B;Yj=run.mm(J,H)
        if len(x)==5:
            Q,K=x[3:];C=run.mm(Q.T,O);D=run.mm(K,C);Yo=run.mm(Q,D);Y=Yj+Yo
            answer={k:host(v) for k,v in dict(w=w,B=B,H=H,C=C,D=D).items()}
        else:
            Y=Yj;answer={k:host(v) for k,v in dict(w=w,B=B,H=H).items()}
        remember(refs,locals())
        return Y,answer
    def dual(x,p,d):
        J,c,O=x[:3];dJ,dc,dO=d[:3]
        dB=dJ.T@O+J.T@dO
        dH=(2*c*dc)[:,None]*p['B']+p['w'][:,None]*dB
        answer=dJ@p['H']+J@dH
        if len(x)==5:
            Q,K=x[3:];dQ,dK=d[3:]
            dC=dQ.T@O+Q.T@dO
            dD=dK@p['C']+K@dC
            answer=answer+dQ@p['D']+Q@dD
        remember(refs,locals())
        return answer
    def compare(x,p,g,cot,d,label):
        tangent=dual(x,p,d)
        analytic=float(np.sum(cot*tangent,dtype=np.float64))
        backward=float(sum(float(np.sum(gi*di,dtype=np.float64)) for gi,di in zip(g,d)))
        absolute=abs(analytic-backward);scale=max(abs(analytic),abs(backward))
        relative=absolute/scale if scale>0 else None
        stable=norm(cot)*norm(tangent)+sum(norm(gi)*norm(di) for gi,di in zip(g,d))
        require(all(np.isfinite(v) for v in (analytic,backward,absolute,scale,stable)) and stable>0,
                'Finite nonzero independent derivative comparison scale')
        answer=dict(direction=label,analytic_dot=analytic,autograd_dot=backward,
            absolute_error=absolute,relative_error=relative,stable_norm_denominator=stable,
            norm_scaled_defect=absolute/stable,
            passed=relative is not None and absolute<=ABS_LIMIT and relative<=REL_LIMIT)
        remember(refs,locals());return answer
    def case(run,arrays,dtype,label,reference_bytes=None,isolated=False):
        x=tuple(tensor(a,dtype,True) for a in arrays)
        before=[hashlib.sha256(raw(a)).hexdigest() for a in x]
        cot0=random((x[0].shape[0],run.rank));cot0=.01*cot0/norm(cot0)
        cot=tensor(cot0,dtype)
        Y=run(*x)
        loss=(Y.to(torch.float64)*cot.to(torch.float64)).sum()
        gradients=torch.autograd.grad(loss,x,create_graph=False,retain_graph=False)
        require(all(bool(torch.isfinite(g).all()) and bool((g!=0).any()) for g in gradients),
                'All connected leaf gradients finite/nonzero')
        with torch.no_grad():reconstructed,p=primals(run,x)
        same=raw(Y)==raw(reconstructed)
        if reference_bytes is not None:same=same and raw(Y)==reference_bytes
        require(same,'Byte-exact rounded primals and saved production Y before independent dual')
        X=tuple(host(v) for v in x);G=tuple(host(v) for v in gradients);COT=host(cot)
        directions=tuple(direction(v.shape,dtype) for v in x)
        tests=[]
        if isolated:
            for index,name in enumerate(('J','lambda','Omega','Qp','core')[:len(x)]):
                one=tuple(directions[j] if j==index else np.zeros_like(directions[j]) for j in range(len(x)))
                tests.append(compare(X,p,G,COT,one,name))
        tests.append(compare(X,p,G,COT,directions,'all_independent_joint'))
        checkpoint_Y=checkpoint(lambda *v:run(*v),*x,use_reentrant=False,preserve_rng_state=False)
        checkpoint_loss=(checkpoint_Y.to(torch.float64)*cot.to(torch.float64)).sum()
        checkpoint_gradients=torch.autograd.grad(checkpoint_loss,x,create_graph=False,retain_graph=False)
        cp_output=raw(Y)==raw(checkpoint_Y)
        cp_grad=[raw(a)==raw(b) for a,b in zip(gradients,checkpoint_gradients)]
        immutable=before==[hashlib.sha256(raw(a)).hexdigest() for a in x]
        answer=dict(case=label,dtype=str(dtype),rank=run.rank,inputs=len(x),
            all_leaf_gradients_finite_nonzero=True,all_leaf_gradients_Frobenius=[norm(v) for v in G],
            reconstructed_Y_byte_exact=same,saved_native_Y_checked=reference_bytes is not None,
            output_sha256=hashlib.sha256(raw(Y)).hexdigest(),
            independent_tangent=tests,checkpoint_output_byte_exact=cp_output,
            checkpoint_gradient_bytes=cp_grad,inputs_unchanged=immutable,cotangent_Frobenius=norm(COT),
            passed=same and all(t['passed'] for t in tests) and cp_output and all(cp_grad) and immutable)
        remember(refs,locals());tick();return answer
    original_product=workspace.product
    calls=[]
    expected=PRODUCTION_SHAPES
    def counted(A,B):
        shape=(int(A.shape[0]),int(A.shape[1]),int(B.shape[1]))
        require(len(calls)<20 and shape==expected[len(calls)%5],
                'At most four exact original five-product forward equivalents')
        calls.append(shape);return original_product(A,B)
    workspace.product=counted
    try:
        math_run=bridge.build_mathematical(torch,prepared);refs.append(weakref.ref(math_run))
        tiny=[random((12,8),.15),random((8,),.2),random((12,3),.3),
              random((12,3),.15),random((3,3),.2)]
        rows=[case(math_run,tiny,torch.float64,'tiny_old_present',isolated=True),
              case(math_run,tiny[:3],torch.float64,'tiny_no_old',isolated=True)]
        require(not calls,'Separate tiny64 branch never uses production workspace')
        def member(name):
            row=bundle['metadata']['arrays'][name]
            answer=np.frombuffer(bundle['blob'],dtype=row['dtype'],
                count=row['bytes']//np.dtype(row['dtype']).itemsize,offset=row['offset']).reshape(row['shape'])
            refs.append(weakref.ref(answer));return answer
        p=member('input.p');previous=member('input.previous');w=member('input.w')
        lam=member('input.lam');Q=member('input.Q');core=member('input.core')
        reference=member('native.Y')
        actual=(p[previous,6144:8128],lam[6144:8128],w[previous,:384],Q[previous,:],core)
        production=bridge.build(torch,np,prepared,bridge.attest(prepared,np,torch,
            __import__('threadpoolctl').threadpool_info,tick),workspace)
        refs.append(weakref.ref(production))
        rows.append(case(production,actual,torch.float32,'production_entry8',
                         reference_bytes=reference.tobytes(),isolated=False))
        require(15<=len(calls)<=20,'Actual plain/reconstruction/checkpoint calls and bounded replay')
        return dict(passed=all(r['passed'] for r in rows),completed=True,unit_cases=3,rows=rows,
            seed=SEED,absolute_limit=ABS_LIMIT,relative_limit=REL_LIMIT,
            cotangent_requested_Frobenius=.01,native_mm_calls=len(calls),native_mm_shapes=calls,
            complete_source_upper=SOURCE_UPPER,hard_flop_cap=100_000_000_000,
            derived_base_F=DERIVED_BASE_F,remaining_tiny_and_scalar_reserve_F=SOURCE_UPPER-DERIVED_BASE_F,
            actual_deployment_bill_claimed=False,real_algebra_linearization_not_rounding_derivative=True,
            no_finite_difference=True,all_five_production_paths_tested=True,
            full_model_calls=0,parameter_updates=0,truth_accessed=False,score_measured=False)
    finally:
        workspace.product=original_product
        remember(refs,locals())
