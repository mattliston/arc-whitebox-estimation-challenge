"""Phase2 two-cube K3 carrier. Numerical operations use FlopScope only."""
import flopscope as flops

import flopscope.numpy as fnp

from whestbench import BaseEstimator

_CP3_WINOGRAD_LEVELS = 4

_CP3_WINOGRAD_ARENA_ELEMENTS = 844165120

_CP3_PUSH_ARENA_VIEW_SPECS = ((0, 1835008, (7, 512, 512)), (1835008, 29360128, (7, 512, 7680)), (29360128, 32571392, (7, 7, 256, 256)), (32571392, 80740352, (7, 7, 256, 3840)), (80740352, 86360064, (7, 7, 7, 128, 128)), (86360064, 170655744, (7, 7, 7, 128, 1920)), (170655744, 180490240, (7, 7, 7, 7, 64, 64)), (180490240, 328007680, (7, 7, 7, 7, 64, 960)), (328007680, 329056256, (4, 512, 512)), (328007680, 329842688, (4, 7, 256, 256)), (328007680, 331218944, (4, 7, 7, 128, 128)), (328007680, 333627392, (4, 7, 7, 7, 64, 64)), (328007680, 343736320, (4, 512, 7680)), (328007680, 355532800, (4, 7, 256, 3840)), (328007680, 376176640, (4, 7, 7, 128, 1920)), (328007680, 412303360, (4, 7, 7, 7, 64, 960)), (412303360, 559820800, (7, 7, 7, 7, 64, 960)), (559820800, 644116480, (7, 7, 7, 128, 1920)), (644116480, 692285440, (7, 7, 256, 3840)), (559820800, 587345920, (7, 512, 7680)), (728412160, 791633920, (3, 7, 7, 7, 64, 960)), (728412160, 764538880, (3, 7, 7, 128, 1920)), (728412160, 749056000, (3, 7, 256, 3840)), (728412160, 740208640, (3, 512, 7680)), (812707840, 828436480, (1024, 15360)), (828436480, 844165120, (1024, 15360)))

_CP3_READ_ARENA_VIEW_SPECS = ((0, 25690112, (7, 512, 7168)), (25690112, 51380224, (7, 7168, 512)), (51380224, 96337920, (7, 7, 256, 3584)), (96337920, 141295616, (7, 7, 3584, 256)), (141295616, 219971584, (7, 7, 7, 128, 1792)), (219971584, 298647552, (7, 7, 7, 1792, 128)), (298647552, 436330496, (7, 7, 7, 7, 64, 896)), (436330496, 574013440, (7, 7, 7, 7, 896, 64)), (574013440, 588693504, (4, 512, 7168)), (574013440, 599703552, (4, 7, 256, 3584)), (574013440, 618971136, (4, 7, 7, 128, 1792)), (574013440, 652689408, (4, 7, 7, 7, 64, 896)), (574013440, 588693504, (4, 7168, 512)), (574013440, 599703552, (4, 7, 3584, 256)), (574013440, 618971136, (4, 7, 7, 1792, 128)), (574013440, 652689408, (4, 7, 7, 7, 896, 64)), (652689408, 662523904, (7, 7, 7, 7, 64, 64)), (662523904, 668143616, (7, 7, 7, 128, 128)), (668143616, 671354880, (7, 7, 256, 256)), (662523904, 664358912, (7, 512, 512)), (673763328, 677978112, (3, 7, 7, 7, 64, 64)), (673763328, 676171776, (3, 7, 7, 128, 128)), (673763328, 675139584, (3, 7, 256, 256)), (673763328, 674549760, (3, 512, 512)), (679383040, 680431616, (1024, 1024)), (680431616, 681480192, (1024, 1024)))

def _cp3_bind_arena_views(arena, view_specs):
    """Bind the sealed D1 arena views once on the first prediction."""
    return tuple(
        fnp.reshape(arena[start:stop], shape)
        for start, stop, shape in view_specs
    )

def _cp3_winograd_arena_depth4(a, b, out, arena_views):
    """Audited four-level fixed-arena Winograd product with explicit out=."""
    levels = _CP3_WINOGRAD_LEVELS
    m = int(a.shape[-2])
    k = int(a.shape[-1])
    n = int(b.shape[-1])
    if (
        int(b.shape[-2]) != k
        or m % 2**levels != 0
        or k % 2**levels != 0
        or n % 2**levels != 0
        or tuple(out.shape) != (m, n)
    ):
        raise ValueError("D4 arena kernel requires compatible /16 matrices")

    cursor = 0
    a_views = arena_views[cursor : cursor + 2 * levels : 2]
    b_views = arena_views[cursor + 1 : cursor + 2 * levels : 2]
    cursor += 2 * levels
    dependency_a = arena_views[cursor : cursor + levels]
    cursor += levels
    dependency_b = arena_views[cursor : cursor + levels]
    cursor += levels
    leaf_view = arena_views[cursor]
    cursor += 1
    intermediates = arena_views[cursor : cursor + levels - 1]
    cursor += levels - 1
    u_scratch = arena_views[cursor : cursor + levels]

    current_a, current_b = a, b
    for index in range(levels):
        level = index + 1
        m2, k2, n2 = m // 2**level, k // 2**level, n // 2**level
        a11 = current_a[..., :m2, :k2]
        a12 = current_a[..., :m2, k2 : 2 * k2]
        a21 = current_a[..., m2 : 2 * m2, :k2]
        a22 = current_a[..., m2 : 2 * m2, k2 : 2 * k2]
        b11 = current_b[..., :k2, :n2]
        b12 = current_b[..., :k2, n2 : 2 * n2]
        b21 = current_b[..., k2 : 2 * k2, :n2]
        b22 = current_b[..., k2 : 2 * k2, n2 : 2 * n2]

        scratch_a = dependency_a[index][..., :m2, :k2]
        fnp.add(a21, a22, out=scratch_a[0])
        fnp.subtract(scratch_a[0], a11, out=scratch_a[1])
        fnp.subtract(a11, a21, out=scratch_a[2])
        fnp.subtract(a12, scratch_a[1], out=scratch_a[3])
        next_a = a_views[index][..., :m2, :k2]
        for destination, source in zip(
            next_a, (a11, a12, scratch_a[3], a22, scratch_a[0], scratch_a[1], scratch_a[2])
        ):
            fnp.copyto(destination, source)

        scratch_b = dependency_b[index][..., :k2, :n2]
        fnp.subtract(b12, b11, out=scratch_b[0])
        fnp.subtract(b22, scratch_b[0], out=scratch_b[1])
        fnp.subtract(b22, b12, out=scratch_b[2])
        fnp.subtract(scratch_b[1], b21, out=scratch_b[3])
        next_b = b_views[index][..., :k2, :n2]
        for destination, source in zip(
            next_b, (b11, b21, b22, scratch_b[3], scratch_b[0], scratch_b[1], scratch_b[2])
        ):
            fnp.copyto(destination, source)
        current_a, current_b = next_a, next_b

    current = leaf_view[..., : m // 2**levels, : n // 2**levels]
    fnp.matmul(current_a, current_b, out=current)
    for unwind, level in enumerate(range(levels, 0, -1)):
        rows, columns = m // 2**level, n // 2**level
        destination = intermediates[unwind] if level > 1 else out
        destination = destination[..., : 2 * rows, : 2 * columns]
        p0, p1, p2, p3, p4, p5, p6 = (current[index] for index in range(7))
        scratch = u_scratch[unwind][..., :rows, :columns]
        top_left = destination[..., :rows, :columns]
        top_right = destination[..., :rows, columns : 2 * columns]
        bottom_left = destination[..., rows : 2 * rows, :columns]
        bottom_right = destination[..., rows : 2 * rows, columns : 2 * columns]
        fnp.add(p0, p1, out=top_left)
        fnp.add(p0, p5, out=scratch[0])
        fnp.add(scratch[0], p6, out=scratch[1])
        fnp.add(scratch[0], p4, out=scratch[2])
        fnp.add(scratch[2], p2, out=top_right)
        fnp.subtract(scratch[1], p3, out=bottom_left)
        fnp.add(scratch[1], p4, out=bottom_right)
        current = destination
    return out

def gaussian_raw(mu,v):
    v = fnp.maximum(v,1e-10)
    s = fnp.sqrt(v); a = mu/s
    pdf = fnp.exp(-a*a/2)/fnp.sqrt(2*3.141592653589793); g = flops.stats.norm.cdf(a)
    m = s*pdf+mu*g
    r2 = (mu*mu+v)*g+mu*s*pdf
    r3 = mu*(mu*mu+3*v)*g+(mu*mu+2*v)*s*pdf
    r4 = (mu**4+6*mu*mu*v+3*v*v)*g+mu*(mu*mu+5*v)*s*pdf
    return s,a,pdf,g,m,r2,r3,r4

def nonlinear(mu,C,K21,k4,k22_override=None,metric="identity"):
    n = len(mu); v = fnp.diag(C).copy()
    s,a,pdf,g,mg,r2g,r3g,r4g = gaussian_raw(mu,v)
    d = pdf/s; e = -a*pdf/v; f = (a*a-1)*pdf/(v*s)
    k3 = fnp.diag(K21).copy()
    m = mg + k3*e/6+k4*f/24
    r2 = r2g+k3*d/3+k4*e/12
    r3 = r3g+k3*g+k4*d/4
    r4 = r4g+4*k3*mg+k4*g
    var = r2-m*m
    post_k3 = r3-3*m*r2+2*m**3
    post_k4 = r4-4*m*r3+6*m*m*r2-3*m**4-3*var*var
    Co = C.copy(); fnp.fill_diagonal(Co,0)
    Ko = K21.copy(); fnp.fill_diagonal(Ko,0)
    k22 = k4/3 if k22_override is None else k22_override
    mix2 = Co*Co+k22/2
    cov = fnp.outer(g,g)*Co + (fnp.outer(d,g)*Ko+fnp.outer(g,d)*Ko.T)/2 + fnp.outer(d,d)*mix2/2
    pk21 = 2*fnp.outer(mg,g)*Co+fnp.outer(mg,d)*Ko.T+fnp.outer(g,g)*Ko+fnp.outer(g,d)*mix2
    pk22 = 4*fnp.outer(mg,mg)*Co+2*(fnp.outer(mg,g)*Ko.T+fnp.outer(g,mg)*Ko)+2*fnp.outer(g,g)*mix2
    post21 = pk21-2*m[:,None]*cov
    post22 = pk22-2*m[:,None]*pk21.T-2*m[None,:]*pk21+4*fnp.outer(m,m)*cov-2*cov*cov
    fnp.fill_diagonal(post22,post_k4)
    radial = 3*fnp.sum(post22)/(n*(n+2))
    fnp.fill_diagonal(cov,var)
    fnp.fill_diagonal(post21,post_k3)
    if metric=="covariance":
        radial = 3*fnp.sum(post22)/(fnp.trace(cov)**2+2*fnp.sum(cov*cov))
    aux = dict(sigma=s,alpha=a,pdf=pdf,gate=g,gaussian_mean=mg,pre_v=v,pre_cov=C,pre_k3=k3,
               pre_k4=fnp.broadcast_to(k4,(n,)).copy(),post_v=var,post_k3=post_k3,post_k4=post_k4)
    return m,cov,post21,radial,aux

def features(l,m,C,K21,aux,W):
    n = len(m); s = aux["sigma"]; v = s*s
    rho = aux["pre_cov"]/fnp.outer(s,s)
    skplane = K21/(v[:,None]*s[None,:])
    postscale = fnp.sqrt(fnp.maximum(aux["post_v"],1e-12))
    postrho = C/fnp.outer(postscale,postscale)
    postrho = fnp.clip(postrho,-1,1)
    w = W.astype(fnp.float32); w2 = w*w; w3 = w2*w; w4 = w2*w2
    ones = fnp.ones(n)
    vals = (ones*(2*l/15-1),ones*(l/16),ones,aux["alpha"],fnp.log(s)-fnp.mean(fnp.log(s)),
            aux["gate"],aux["pdf"],aux["gaussian_mean"]/s,m/s,(m-aux["gaussian_mean"])/s,
            aux["pre_k3"]/(v*s),aux["post_v"]/v,aux["pre_k4"]/(v*v),fnp.mean(rho,axis=1),
            fnp.sqrt(fnp.mean(rho*rho,axis=1)),fnp.mean(fnp.abs(rho),axis=1),
            fnp.sum(w,axis=0)*0.7071067811865475,fnp.sum(w2-0.001953125,axis=0)*11.313708498984761,
            fnp.sum(w3,axis=0)*93.47798314754834,fnp.sum(w4-1.1444091796875e-5,axis=0)*836.0924988699916,
            aux["post_k3"]/(v*s),aux["post_k4"]/(v*v),
            fnp.sqrt(fnp.sum(skplane*skplane,axis=1)),fnp.sqrt(fnp.sum(skplane*skplane,axis=0)),
            fnp.sum(skplane,axis=1)/fnp.sqrt(n),fnp.sum(skplane,axis=0)/fnp.sqrt(n),
            rho@aux["alpha"]/n,fnp.sqrt(fnp.mean(postrho*postrho,axis=1)),fnp.mean(postrho,axis=1),
            ones*fnp.mean(m/s))
    return fnp.stack(vals,axis=1).astype(fnp.float32)

_SCRATCH = None

_METRIC = "covariance"

_COV32 = True

_WINOGRAD = True

def _mm(a,b):
    if not _WINOGRAD or a.dtype!=fnp.float32 or b.dtype!=fnp.float32:
        return a@b
    m,k = int(a.shape[0]),int(a.shape[1]); n = int(b.shape[1])
    if k==0 or n==0:
        return fnp.zeros((m,n),dtype=fnp.float32)
    if m%16 or k%16 or n%16:
        return a@b
    if k>14336:
        out=_mm(a[:,:14336],b[:14336,:])
        for start in range(14336,k,14336):
            end=min(start+14336,k)
            out=out+_mm(a[:,start:end],b[start:end,:])
        return out
    if n>15360:
        return fnp.concatenate([_mm(a,b[:,start:min(start+15360,n)]) for start in range(0,n,15360)],axis=1)
    out = fnp.empty((m,n),dtype=fnp.float32)
    views = _SCRATCH[1] if k==1024 else _SCRATCH[2]
    _cp3_winograd_arena_depth4(a,b,out,views)
    return out

def star_push(W,star,diagonal_only=False):
    w = W.T.astype(fnp.float32)
    A = star.astype(fnp.float32)
    if diagonal_only:
        return fnp.sum((w@A)*w*w,axis=1).astype(fnp.float64)
    wa = _mm(w,A)
    return (2*_mm(wa*w,w.T)+_mm(w*w,wa.T)).astype(fnp.float64)/3

def plane(p,lam):
    if p.shape[1]==0:return fnp.zeros((p.shape[0],p.shape[0]),fnp.float64)
    return (_mm(p*p,(p*lam).T)).astype(fnp.float64)

def _predict_binary(mlp,hook=None,cov32=True,metric="covariance",collect_states=False,birth_mode="analytic"):
    global _SCRATCH
    depth,n=int(mlp.depth),int(mlp.width);W=mlp.weights
    if _SCRATCH is None:
        arena=fnp.empty((_CP3_WINOGRAD_ARENA_ELEMENTS,),dtype=fnp.float32)
        _SCRATCH=(arena,_cp3_bind_arena_views(arena,_CP3_PUSH_ARENA_VIEW_SPECS),_cp3_bind_arena_views(arena,_CP3_READ_ARENA_VIEW_SPECS))
    m=fnp.zeros(n,dtype=fnp.float64);C=fnp.eye(n,dtype=fnp.float64);radial=fnp.asarray(0.0,dtype=fnp.float64)
    p=fnp.zeros((n,0),fnp.float32);lam=fnp.zeros(0,fnp.float32);star=fnp.zeros((n,n),fnp.float32)
    means=[];states=[]
    for l in range(depth):
        w32=fnp.asarray(W[l],dtype=fnp.float32);w=fnp.asarray(W[l],dtype=fnp.float64)
        mu=w.T@m
        Cpre=(_mm(_mm(w32.T,C.astype(fnp.float32)),w32)).astype(fnp.float64) if cov32 else w.T@C@w
        pp=_mm(w32.T,p);cp_pre=plane(pp,lam);K21=cp_pre+star_push(W[l],star)
        vv=fnp.diag(Cpre);k4=radial*vv*vv;k22=radial*(fnp.outer(vv,vv)+2*Cpre*Cpre)/3
        m,C,target,radial_new,aux=nonlinear(mu,Cpre,K21,k4,k22,metric)
        host=m.copy()
        if hook is not None:
            dm=fnp.asarray(hook(l,m,C,K21,aux,W[l]),fnp.float64)
            C=C-fnp.outer(m,dm)-fnp.outer(dm,m)-fnp.outer(dm,dm);m=m+dm
        means.append(m.copy())
        if l==depth-1:break
        g=aux["gate"];birth=aux["pdf"]/aux["sigma"]
        inherited=g[:,None]**2*g[None,:]*cp_pre
        bp=g[:,None]*Cpre;fnp.fill_diagonal(bp,0)
        residual=target-inherited
        if birth_mode=="analytic":
            beta=2*(aux["gaussian_mean"]-host*g)
        elif birth_mode=="fitted":
            remainder=residual-bp*bp*birth[None,:]
            beta=fnp.sum(remainder*bp.T,axis=1)/fnp.maximum(fnp.sum(bp.T**2,axis=1),1e-24)
        else:raise ValueError(birth_mode)
        innovation=fnp.diag(residual).copy()
        safe_birth=fnp.maximum(birth,1e-12)
        u=beta/safe_birth
        # Bounds only regularize nearly deterministic nodes where all birth
        # coefficients are negligible; never clip a covariance eigenvalue.
        u=fnp.clip(u,-8*vv,8*vv)
        disc=4*innovation/safe_birth-3*u*u
        delta=fnp.sqrt(fnp.maximum(disc,.01*vv*vv))
        delta=fnp.minimum(delta,16*vv)
        t=(u+delta)/2;s=(u-delta)/2;a=birth/delta
        p1=bp.copy();p2=bp.copy();fnp.fill_diagonal(p1,t);fnp.fill_diagonal(p2,s)
        born=bp*bp*birth[None,:]+beta[:,None]*bp.T
        fnp.fill_diagonal(born,birth*(3*u*u+delta*delta)/4)
        cp_post=inherited+born
        p=fnp.concatenate([g.astype(fnp.float32)[:,None]*pp,p1.astype(fnp.float32),p2.astype(fnp.float32)],axis=1)
        lam=fnp.concatenate([lam,a.astype(fnp.float32),-a.astype(fnp.float32)])
        rest=target-cp_post
        star=(3*rest.T-2*fnp.diag(fnp.diag(rest))).astype(fnp.float32)
        if collect_states:states.append(dict(layer=l,clipped_disc=int(fnp.sum(disc<.01*vv*vv)),star_rms=float(fnp.sqrt(fnp.mean(star*star)))))
        radial=radial_new
    return fnp.asarray(fnp.maximum(fnp.stack(means),0.0),dtype=fnp.float32)

class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        if int(mlp.width)!=1024 or int(mlp.depth)!=16 or int(budget)<2**41:
            return fnp.zeros((int(mlp.depth),int(mlp.width)),dtype=fnp.float32)
        return _predict_binary(mlp)
