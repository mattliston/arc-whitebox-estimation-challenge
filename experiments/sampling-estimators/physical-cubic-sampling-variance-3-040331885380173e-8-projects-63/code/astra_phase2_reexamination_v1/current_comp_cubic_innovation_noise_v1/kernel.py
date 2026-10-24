"""Fixed target-free cubic-innovation tangent-noise circuit; inert definitions.

FNP is supplied by the admitted NumPy-free RPC worker. No NumPy, full
estimator/head execution, labels, fitted coefficients or automatic next stage.
"""
import hashlib

N, WIDTH, LAYER = 32, 1024, 8
SEED = 202609130835
CAP = 14_000_000_000
PRECISION = 1e-9
SECTIONS = ('ingress', 'random_and_interpolation', 'prefix_and_tangents',
            'query_and_current_response', 'replica_0_nonlinear', 'replica_0_linear',
            'replica_1_nonlinear', 'replica_1_linear', 'noise_statistics_and_exports')
SECTION_CAPS = (250_000_000, 20_000_000, 8_300_000_000, 25_000_000,
                975_000_000, 975_000_000, 975_000_000, 975_000_000, 25_000_000)
DENSE_ONE = 2*N*WIDTH*(2*WIDTH-1)
GEMM_COUNTS = dict(prefix_and_tangents=61,
    replica_0_nonlinear=7, replica_0_linear=7,
    replica_1_nonlinear=7, replica_1_linear=7)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def cost_certificate():
    """Scalar source certificate; actual meter, never this estimate, is billed."""
    require(sum(SECTION_CAPS) < CAP, 'Prospective complete allowance below14BF')
    return dict(outer=N, width=WIDTH, prefix_layers=9, tangent_replicas=2,
        prefix_matmuls=27, tangent_matmuls=34, query_matmuls=28,
        dense_FP64_one=DENSE_ONE, dense_FP64_total=89*DENSE_ONE,
        section_caps=dict(zip(SECTIONS, SECTION_CAPS)),
        complete_source_allowance=sum(SECTION_CAPS), hard_cap=CAP,
        billed_dtype='float64', raw_weight_dtype='float32',
        whole_estimator_cost=False, projected_tensor_materialized=False)


def run(fnp, load_serialized, section, own, ctx):
    """All construction, arithmetic, scans, copies and reductions are paid.

    Only the admitted byte loader enters. Returned values are plain metadata;
    no numerical proxy is retained by this module or the caller's namespace.
    """
    calls = {key:0 for key in GEMM_COUNTS}
    def array(value):
        return own(value)
    def scalar(value):
        return array(fnp.full((1,), value, dtype=fnp.float64))
    def finite(value):
        require(bool(fnp.all(fnp.isfinite(value))), 'Finite actual paid circuit value')
    def mm(left, right, group):
        require(tuple(left.shape)==(N,WIDTH) and tuple(right.shape)==(WIDTH,WIDTH)
            and str(left.dtype)==str(right.dtype)=='float64', 'Exact FP64 batch matrix geometry')
        calls[group] += 1
        before=int(ctx.flops_used)
        result=array(left@right)
        require(int(ctx.flops_used)-before==DENSE_ONE, 'Actual exact FP64 dense-product tariff')
        return result
    def export(value):
        copied=array(value.astype(fnp.float64))
        finite(copied)
        raw, shape, dtype=copied._fetch_data()
        require(tuple(shape)==tuple(value.shape) and str(dtype)=='float64'
            and len(raw)==8*value.size, 'Detached paid FP64 representation')
        return dict(shape=list(shape), dtype='float64', bytes=len(raw),
                    raw_sha256=hashlib.sha256(raw).hexdigest())
    def ingress():
        serialized=load_serialized()
        weights=array(fnp.frombuffer(serialized['weights'],dtype=fnp.float32)
            .reshape((10,WIDTH,WIDTH)).astype(fnp.float64))
        values={}
        for name in ('alive','gate','alpha','pdf','sigma'):
            dtype=fnp.bool_ if name=='alive' else fnp.float64
            value=array(fnp.frombuffer(serialized[name],dtype=dtype)
                .reshape((WIDTH,)).astype(fnp.float64))
            finite(value)
            values[name]=value
        finite(weights)
        zero, one, two, six = (scalar(x) for x in (0.,1.,2.,6.))
        require(bool(fnp.all(values['sigma']>zero))
            and bool(fnp.all(values['pdf']>=zero))
            and bool(fnp.all(values['gate']>=zero))
            and bool(fnp.all(values['gate']<=one)), 'Captured response domains')
        return weights,values,(zero,one,two,six)
    weights,values,(zero,one,two,six)=section('ingress',ingress)
    def random_inputs():
        rng=fnp.random.default_rng(SEED)
        X=array(rng.standard_normal((N,WIDTH)))
        Z=array(rng.standard_normal((N,WIDTH)))
        U=array(rng.standard_normal((N,WIDTH)))
        E0=array(rng.standard_normal((N,WIDTH)))
        E1=array(rng.standard_normal((N,WIDTH)))
        t=array(rng.uniform(0.,1.,(N,1)))
        Y=array(t*X+fnp.sqrt(one-t*t)*Z)
        for item in (X,Z,U,E0,E1,t,Y):
            finite(item)
        return X,Y,U,E0,E1
    X,Y,U,E0,E1=section('random_and_interpolation',random_inputs)
    def prefix():
        group='prefix_and_tangents'
        def forward(value,keep_gates):
            gates=[]
            for layer in range(9):
                value=mm(value,weights[layer],group)
                finite(value)
                if layer<8:
                    if keep_gates:
                        gates.append(array(value>zero))
                    value=array(fnp.maximum(value,zero))
            return value,gates
        AX,gx=forward(X,True)
        AY,gy=forward(Y,True)
        AU,unused=forward(U,False)
        tangents=[]
        for direction in (E0,E1):
            # The derivative through the first linear map is identical for X/Y.
            first=mm(direction,weights[0],group)
            tx=array(first*gx[0]); ty=array(first*gy[0])
            for layer in range(1,9):
                tx=mm(tx,weights[layer],group)
                ty=mm(ty,weights[layer],group)
                if layer<8:
                    tx=array(tx*gx[layer]); ty=array(ty*gy[layer])
            finite(tx);finite(ty)
            tangents.append((tx,ty))
        return AX,AY,AU,tangents
    AX,AY,AU,tangents=section('prefix_and_tangents',prefix)
    def response():
        Q=weights[9]
        Q2=array(Q*Q);Q3=array(Q2*Q)
        alive=values['alive']
        G=array(alive*values['gate'])
        ah=array(alive*(fnp.maximum(AY,zero)-fnp.maximum(AU,zero)))
        ag=array(G*(AY-AU))
        hx=array(alive*(AX>zero));hy=array(alive*(AY>zero))
        s=array(-values['alpha']*values['pdf']/(six*values['sigma']*values['sigma']))
        for item in (Q2,Q3,G,ah,ag,hx,hy,s):
            finite(item)
        return Q,Q2,Q3,G,ah,ag,hx,hy,s
    Q,Q2,Q3,G,ah,ag,hx,hy,s=section('query_and_current_response',response)
    def query(a,b,c,group):
        # 2*SymAvg(a,b,c), exactly de-repeated on coordinate indices.
        qa=mm(a,Q,group);qb=mm(b,Q,group);qc=mm(c,Q,group)
        ab=array(a*b);ac=array(a*c);bc=array(b*c)
        qab=mm(ab,Q2,group);qac=mm(ac,Q2,group);qbc=mm(bc,Q2,group)
        qabc=mm(array(ab*c),Q3,group)
        result=array(two*(qa*qb*qc-qab*qc-qac*qb-qbc*qa+two*qabc))
        finite(result)
        return result
    queried=[]
    for replica,(tx,ty) in enumerate(tangents):
        hn='replica_'+str(replica)+'_nonlinear'
        ln='replica_'+str(replica)+'_linear'
        h=section(hn,lambda:query(ah,array(hx*tx),array(hy*ty),hn))
        g=section(ln,lambda:query(ag,array(G*tx),array(G*ty),ln))
        queried.append((h,g))
    def statistics():
        q0=array(queried[0][0]-queried[0][1])
        q1=array(queried[1][0]-queried[1][1])
        nrows=scalar(float(N));nwidth=scalar(float(WIDTH))
        host0=array(s*fnp.sum(q0,axis=0)/nrows)
        host1=array(s*fnp.sum(q1,axis=0)/nrows)
        mean_difference=array(host0-host1)
        row_difference=array((q0-q1)*s)
        paired_mse=fnp.sum(mean_difference*mean_difference)/nwidth
        v32=fnp.sum(row_difference*row_difference)/(two*nrows*nrows*nwidth)
        for value in (host0,host1,mean_difference,row_difference,paired_mse,v32):
            finite(value)
        primary=dict(paired_average_difference_mse=float(paired_mse),
            rowwise_tangent_variance_V32=float(v32),threshold=PRECISION,
            paired_average_passed=bool(paired_mse<=scalar(PRECISION)),
            rowwise_variance_passed=bool(v32<=scalar(PRECISION)))
        primary['host_noise_precision_passed']=(primary['paired_average_passed']
            and primary['rowwise_variance_passed'])
        return dict(name='fixed32_two_tangent_noise',primary=primary,
            exports=dict(replica0_host_average=export(host0),replica1_host_average=export(host1),
                paired_average_difference=export(mean_difference),
                scaled_rowwise_difference=export(row_difference)),
            current_gate_includes_alive=True,hard_pathwise_gates_include_alive=True,
            raw_mean_or_final_score_measured=False,outer_variance_measured=False)
    row=section('noise_statistics_and_exports',statistics)
    require(calls==GEMM_COUNTS, 'Exactly61 prefix/tangent and28 diagonal-query products')
    row['matmul_counts']=calls
    row['matmul_flops']=89*DENSE_ONE
    row['seed']=SEED
    row['outer']=N
    row['independent_tangent_replicas']=2
    row['physical_prefix_forward_passes']=3
    row['physical_tangent_prefix_passes']=4
    return [row]
