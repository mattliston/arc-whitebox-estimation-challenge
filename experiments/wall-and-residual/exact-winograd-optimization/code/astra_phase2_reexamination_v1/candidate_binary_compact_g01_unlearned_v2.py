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
    if not _WINOGRAD or str(a.dtype)!="float32" or str(b.dtype)!="float32":
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

def _split_views_3(views):
    cursor = 0
    a_views = views[cursor : cursor + 2 * 3 : 2]
    b_views = views[cursor + 1 : cursor + 2 * 3 : 2]
    cursor += 2 * 3
    dependency_a = views[cursor : cursor + 3]
    cursor += 3
    dependency_b = views[cursor : cursor + 3]
    cursor += 3
    leaf = views[cursor]
    cursor += 1
    intermediates = views[cursor : cursor + 3 - 1]
    cursor += 3 - 1
    u_scratch = views[cursor : cursor + 3]
    cursor += 3
    if cursor != len(views):
        raise AssertionError((cursor, len(views)))
    return (
        a_views,
        b_views,
        dependency_a,
        dependency_b,
        leaf,
        intermediates,
        u_scratch,
    )

def _packed_product_3(a, b, out, arena_views):
    """Audited five-level fixed-arena Winograd product with explicit out=."""
    (
        a_views,
        b_views,
        dependency_a,
        dependency_b,
        leaf_view,
        intermediates,
        u_scratch,
    ) = _split_views_3(arena_views)
    levels = 3
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
        raise ValueError("packed D5 requires compatible /32 matrices")

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
            next_a,
            (a11, a12, scratch_a[3], a22, scratch_a[0], scratch_a[1], scratch_a[2]),
        ):
            fnp.copyto(destination, source)

        scratch_b = dependency_b[index][..., :k2, :n2]
        fnp.subtract(b12, b11, out=scratch_b[0])
        fnp.subtract(b22, scratch_b[0], out=scratch_b[1])
        fnp.subtract(b22, b12, out=scratch_b[2])
        fnp.subtract(scratch_b[1], b21, out=scratch_b[3])
        next_b = b_views[index][..., :k2, :n2]
        for destination, source in zip(
            next_b,
            (b11, b21, b22, scratch_b[3], scratch_b[0], scratch_b[1], scratch_b[2]),
        ):
            fnp.copyto(destination, source)
        current_a, current_b = next_a, next_b

    current = leaf_view[..., : m // 2**levels, : n // 2**levels]
    fnp.matmul(current_a, current_b, out=current)
    for unwind, level in enumerate(range(levels, 0, -1)):
        rows, columns = m // 2**level, n // 2**level
        destination = intermediates[unwind] if level > 1 else out
        destination = destination[..., : 2 * rows, : 2 * columns]
        p0, p1, p2, p3, p4, p5, p6 = (
            current[index] for index in range(7)
        )
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

def _split_views_5(views):
    cursor = 0
    a_views = views[cursor : cursor + 2 * 5 : 2]
    b_views = views[cursor + 1 : cursor + 2 * 5 : 2]
    cursor += 2 * 5
    dependency_a = views[cursor : cursor + 5]
    cursor += 5
    dependency_b = views[cursor : cursor + 5]
    cursor += 5
    leaf = views[cursor]
    cursor += 1
    intermediates = views[cursor : cursor + 5 - 1]
    cursor += 5 - 1
    u_scratch = views[cursor : cursor + 5]
    cursor += 5
    if cursor != len(views):
        raise AssertionError((cursor, len(views)))
    return (
        a_views,
        b_views,
        dependency_a,
        dependency_b,
        leaf,
        intermediates,
        u_scratch,
    )

def _packed_product_5(a, b, out, arena_views):
    """Audited five-level fixed-arena Winograd product with explicit out=."""
    (
        a_views,
        b_views,
        dependency_a,
        dependency_b,
        leaf_view,
        intermediates,
        u_scratch,
    ) = _split_views_5(arena_views)
    levels = 5
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
        raise ValueError("packed D5 requires compatible /32 matrices")

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
            next_a,
            (a11, a12, scratch_a[3], a22, scratch_a[0], scratch_a[1], scratch_a[2]),
        ):
            fnp.copyto(destination, source)

        scratch_b = dependency_b[index][..., :k2, :n2]
        fnp.subtract(b12, b11, out=scratch_b[0])
        fnp.subtract(b22, scratch_b[0], out=scratch_b[1])
        fnp.subtract(b22, b12, out=scratch_b[2])
        fnp.subtract(scratch_b[1], b21, out=scratch_b[3])
        next_b = b_views[index][..., :k2, :n2]
        for destination, source in zip(
            next_b,
            (b11, b21, b22, scratch_b[3], scratch_b[0], scratch_b[1], scratch_b[2]),
        ):
            fnp.copyto(destination, source)
        current_a, current_b = next_a, next_b

    current = leaf_view[..., : m // 2**levels, : n // 2**levels]
    fnp.matmul(current_a, current_b, out=current)
    for unwind, level in enumerate(range(levels, 0, -1)):
        rows, columns = m // 2**level, n // 2**level
        destination = intermediates[unwind] if level > 1 else out
        destination = destination[..., : 2 * rows, : 2 * columns]
        p0, p1, p2, p3, p4, p5, p6 = (
            current[index] for index in range(7)
        )
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

_PACKED_ELEMENTS=511393792

_PACKED_PUSH=((0, 1835008, (7, 512, 512)), (17210368, 31890432, (7, 512, 4096)), (154893312, 158104576, (7, 7, 256, 256)), (164727808, 190417920, (7, 7, 256, 2048)), (0, 5619712, (7, 7, 7, 128, 128)), (17210368, 62168064, (7, 7, 7, 128, 1024)), (154893312, 164727808, (7, 7, 7, 7, 64, 64)), (164727808, 243403776, (7, 7, 7, 7, 64, 512)), (0, 17210368, (7, 7, 7, 7, 7, 32, 32)), (17210368, 154893312, (7, 7, 7, 7, 7, 32, 256)), (243403776, 244452352, (4, 512, 512)), (243403776, 245238784, (4, 7, 256, 256)), (243403776, 246615040, (4, 7, 7, 128, 128)), (243403776, 249023488, (4, 7, 7, 7, 64, 64)), (243403776, 253238272, (4, 7, 7, 7, 7, 32, 32)), (243403776, 251792384, (4, 512, 4096)), (243403776, 258083840, (4, 7, 256, 2048)), (243403776, 269093888, (4, 7, 7, 128, 1024)), (243403776, 288361472, (4, 7, 7, 7, 64, 512)), (243403776, 322079744, (4, 7, 7, 7, 7, 32, 256)), (184396800, 322079744, (7, 7, 7, 7, 7, 32, 256)), (0, 78675968, (7, 7, 7, 7, 64, 512)), (184396800, 229354496, (7, 7, 7, 128, 1024)), (0, 25690112, (7, 7, 256, 2048)), (184396800, 199076864, (7, 512, 4096)), (78675968, 137682944, (3, 7, 7, 7, 7, 32, 256)), (78675968, 112394240, (3, 7, 7, 7, 64, 512)), (78675968, 97943552, (3, 7, 7, 128, 1024)), (78675968, 89686016, (3, 7, 256, 2048)), (78675968, 84967424, (3, 512, 4096)))

_PACKED_READ=((0, 14680064, (7, 512, 4096)), (137682944, 152363008, (7, 4096, 512)), (275365888, 301056000, (7, 7, 256, 2048)), (354041856, 379731968, (7, 7, 2048, 256)), (0, 44957696, (7, 7, 7, 128, 1024)), (137682944, 182640640, (7, 7, 7, 1024, 128)), (275365888, 354041856, (7, 7, 7, 7, 64, 512)), (354041856, 432717824, (7, 7, 7, 7, 512, 64)), (0, 137682944, (7, 7, 7, 7, 7, 32, 256)), (137682944, 275365888, (7, 7, 7, 7, 7, 256, 32)), (432717824, 441106432, (4, 512, 4096)), (432717824, 447397888, (4, 7, 256, 2048)), (432717824, 458407936, (4, 7, 7, 128, 1024)), (432717824, 477675520, (4, 7, 7, 7, 64, 512)), (432717824, 511393792, (4, 7, 7, 7, 7, 32, 256)), (432717824, 441106432, (4, 4096, 512)), (432717824, 447397888, (4, 7, 2048, 256)), (432717824, 458407936, (4, 7, 7, 1024, 128)), (432717824, 477675520, (4, 7, 7, 7, 512, 64)), (432717824, 511393792, (4, 7, 7, 7, 7, 256, 32)), (494183424, 511393792, (7, 7, 7, 7, 7, 32, 32)), (0, 9834496, (7, 7, 7, 7, 64, 64)), (494183424, 499803136, (7, 7, 7, 128, 128)), (0, 3211264, (7, 7, 256, 256)), (494183424, 496018432, (7, 512, 512)), (9834496, 17210368, (3, 7, 7, 7, 7, 32, 32)), (9834496, 14049280, (3, 7, 7, 7, 64, 64)), (9834496, 12242944, (3, 7, 7, 128, 128)), (9834496, 11210752, (3, 7, 256, 256)), (9834496, 10620928, (3, 512, 512)))

_PACKED_SQUARE=((0, 1835008, (7, 512, 512)), (5619712, 7454720, (7, 512, 512)), (11239424, 14450688, (7, 7, 256, 256)), (14450688, 17661952, (7, 7, 256, 256)), (0, 5619712, (7, 7, 7, 128, 128)), (5619712, 11239424, (7, 7, 7, 128, 128)), (17661952, 18710528, (4, 512, 512)), (17661952, 19496960, (4, 7, 256, 256)), (17661952, 20873216, (4, 7, 7, 128, 128)), (17661952, 18710528, (4, 512, 512)), (17661952, 19496960, (4, 7, 256, 256)), (17661952, 20873216, (4, 7, 7, 128, 128)), (15253504, 20873216, (7, 7, 7, 128, 128)), (0, 3211264, (7, 7, 256, 256)), (15253504, 17088512, (7, 512, 512)), (3211264, 5619712, (3, 7, 7, 128, 128)), (3211264, 4587520, (3, 7, 256, 256)), (3211264, 3997696, (3, 512, 512)))

_PACKED_TILE=8192

def _mm(a,b):
    if not _WINOGRAD or str(a.dtype)!="float32" or str(b.dtype)!="float32":
        return a@b
    m,k=int(a.shape[0]),int(a.shape[1]);n=int(b.shape[1])
    if k==0 or n==0:return fnp.zeros((m,n),dtype=fnp.float32)
    if n>_PACKED_TILE:
        return fnp.concatenate([_mm(a,b[:,start:min(start+_PACKED_TILE,n)]) for start in range(0,n,_PACKED_TILE)],axis=1)
    if k>_PACKED_TILE:
        out=_mm(a[:,:_PACKED_TILE],b[:_PACKED_TILE,:])
        for start in range(_PACKED_TILE,k,_PACKED_TILE):
            end=min(start+_PACKED_TILE,k)
            out=out+_mm(a[:,start:end],b[start:end,:])
        return out
    out=fnp.empty((m,n),dtype=fnp.float32)
    if m==k==n==1024:
        return _packed_product_3(a,b,out,_SCRATCH[3])
    if m%32 or k%32 or n%32:return a@b
    return _packed_product_5(a,b,out,_SCRATCH[1] if k==1024 else _SCRATCH[2])


def _mm(a,b):
    if not _WINOGRAD or str(a.dtype)!="float32" or str(b.dtype)!="float32":
        return a@b
    m,k=int(a.shape[0]),int(a.shape[1]);n=int(b.shape[1])
    if m==0 or k==0 or n==0:return fnp.zeros((m,n),dtype=fnp.float32)
    if n>_PACKED_TILE:
        return fnp.concatenate([_mm(a,b[:,j:min(j+_PACKED_TILE,n)]) for j in range(0,n,_PACKED_TILE)],axis=1)
    if k>_PACKED_TILE:
        out=_mm(a[:,:_PACKED_TILE],b[:_PACKED_TILE,:])
        for j in range(_PACKED_TILE,k,_PACKED_TILE):
            end=min(j+_PACKED_TILE,k)
            out=out+_mm(a[:,j:end],b[j:end,:])
        return out
    if m%32 or k%32 or n%32:return a@b
    if max(m,k,n)<=1024:
        if min(m,k,n)<=256:return a@b
        out=fnp.empty((m,n),dtype=fnp.float32)
        return _packed_product_3(a,b,out,_SCRATCH[3])
    out=fnp.empty((m,n),dtype=fnp.float32)
    return _packed_product_5(a,b,out,_SCRATCH[1] if n>1024 else _SCRATCH[2])

def _cov_compact(w,C,indices,alive):
    wa=w[indices,:]
    ca=C[fnp.ix_(indices,indices)].astype(fnp.float32)
    result=_mm(wa.T,_mm(ca,wa))
    if int(indices.shape[0])<int(w.shape[0]):
        cold=fnp.nonzero(~alive)[0]
        wc=w[cold,:]
        diagonal=fnp.diag(C)[cold].astype(fnp.float32)
        result=result+_mm((wc*diagonal[:,None]).T,wc)
    return result.astype(fnp.float64)

def _star_compact(w,A,previous,previous_alive,current):
    n=int(w.shape[1])
    wp=w[previous,:]
    aa=A[fnp.ix_(previous,previous)].astype(fnp.float32)
    pushed=_mm(wp.T,aa)
    wr=wp[:,current].T
    pr=pushed[current,:]
    sub=(2*_mm(pr*wr,wr.T)+_mm(wr*wr,pr.T))/3
    diagonal=fnp.sum(pushed*wp.T*wp.T,axis=1,dtype=fnp.float64)
    if int(previous.shape[0])<int(w.shape[0]):
        cold=fnp.nonzero(~previous_alive)[0]
        wc=w[cold,:]
        ad=fnp.diag(A)[cold].astype(fnp.float32)
        wc_current=wc[:,current]
        sub=sub+_mm((wc_current*wc_current).T,wc_current*ad[:,None])
        diagonal=diagonal+fnp.sum((wc*wc*wc*ad[:,None]),axis=0,dtype=fnp.float64)
    result=fnp.zeros((n,n),dtype=fnp.float64)
    fnp.put(result,current[:,None]*n+current[None,:],sub)
    fnp.fill_diagonal(result,diagonal)
    return result


def select(score, floor):
    count = min(len(score), ((int(fnp.nonzero(score >= floor)[0].shape[0])+31)//32)*32)
    indices = fnp.argsort(-score, kind="stable")[:count]
    mask = fnp.zeros(len(score), dtype=fnp.bool_)
    fnp.put(mask,indices,True)
    return indices, mask

def _predict_binary(mlp, hook=None, cov32=True, metric="covariance", collect_states=False, gate_floor=0.01):
    global _SCRATCH
    depth,n=int(mlp.depth),int(mlp.width);W=mlp.weights
    if _SCRATCH is None:
        arena=fnp.empty((_PACKED_ELEMENTS,),dtype=fnp.float32)
        _SCRATCH=(arena,_cp3_bind_arena_views(arena,_PACKED_PUSH),_cp3_bind_arena_views(arena,_PACKED_READ),_cp3_bind_arena_views(arena,_PACKED_SQUARE))
    previous=fnp.arange(n)
    previous_alive=fnp.ones(n,dtype=fnp.bool_)
    m = fnp.zeros(n,dtype=fnp.float64)
    C = fnp.eye(n)
    radial = 0.0
    p = fnp.zeros((n, 0), fnp.float32)
    lam = fnp.zeros(0, fnp.float32)
    star = fnp.zeros((n, n), fnp.float32)
    means, states = [], []
    for l in range(depth):
        w32 = fnp.asarray(W[l],dtype=fnp.float32)
        w = fnp.asarray(W[l],dtype=fnp.float64)
        mu = w.T @ m
        Cpre = _cov_compact(w32,C,previous,previous_alive) if cov32 else w.T @ C @ w
        vv = fnp.diag(Cpre)
        live_indices, alive = select(flops.stats.norm.cdf(mu/fnp.sqrt(fnp.maximum(vv, 1e-10))), gate_floor)
        live = alive[:, None]*alive[None, :]
        pp = _mm(w32[previous,:].T,p[previous,:])
        cp_pre = fnp.zeros((n, n))
        if len(lam):
            sub = pp[live_indices]
            fnp.put(cp_pre,live_indices[:,None]*n+live_indices[None,:],plane(sub,lam))
            diagonal = fnp.sum(pp.astype(fnp.float64)**3*lam.astype(fnp.float64), axis=1)
            fnp.fill_diagonal(cp_pre, diagonal)
        star_pre = _star_compact(w32,star,previous,previous_alive,live_indices)
        K21 = cp_pre+star_pre
        k4 = radial*vv*vv
        k22 = radial*(fnp.outer(vv, vv)+2*Cpre*Cpre)/3
        m, C, target, radial_new, aux = nonlinear(mu, Cpre, K21, k4, k22, metric)
        host = m.copy()
        if hook is not None:
            dm = fnp.asarray(hook(l, m, C, K21, aux, W[l]), fnp.float64)
            C = C-fnp.outer(m, dm)-fnp.outer(dm, m)-fnp.outer(dm, dm)
            m = m+dm
        means.append(m.copy())
        if l == depth-1:
            break
        g = aux["gate"]
        birth = aux["pdf"]/aux["sigma"]
        selected, births = select(fnp.minimum(g, 1-g), gate_floor)
        cv = fnp.diag(C).copy()
        C = C*live
        fnp.fill_diagonal(C, cv)
        td = fnp.diag(target).copy()
        target = target*live
        fnp.fill_diagonal(target, td)
        inherited = g[:, None]**2*g[None, :]*cp_pre
        bp = g[:, None]*Cpre
        fnp.fill_diagonal(bp, 0)
        residual = target-inherited
        beta = 2*(aux["gaussian_mean"]-host*g)
        innovation = fnp.diag(residual).copy()
        safe_birth = fnp.maximum(birth, 1e-12)
        u = fnp.clip(beta/safe_birth, -8*vv, 8*vv)
        disc = 4*innovation/safe_birth-3*u*u
        delta = fnp.minimum(fnp.sqrt(fnp.maximum(disc, .01*vv*vv)), 16*vv)
        t = (u+delta)/2
        s = (u-delta)/2
        a = birth/delta
        p1, p2 = bp.copy(), bp.copy()
        fnp.fill_diagonal(p1, t)
        fnp.fill_diagonal(p2, s)
        born = bp*bp*(birth*births)[None, :]+(beta*births)[:, None]*bp.T
        fnp.fill_diagonal(born, birth*births*(3*u*u+delta*delta)/4)
        cp_post = (inherited+born)*live
        p = fnp.concatenate([g.astype(fnp.float32)[:, None]*pp,
                            p1[:, selected].astype(fnp.float32), p2[:, selected].astype(fnp.float32)], axis=1)
        p = alive.astype(fnp.float32)[:, None]*p
        lam = fnp.concatenate([lam, a[selected].astype(fnp.float32), -a[selected].astype(fnp.float32)])
        rest = target-cp_post
        star = (3*rest.T-2*fnp.diag(fnp.diag(rest))).astype(fnp.float32)
        if collect_states:
            states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1]))
        previous,previous_alive=live_indices,alive
        radial = radial_new
    return fnp.asarray(fnp.maximum(fnp.stack(means),0.0),dtype=fnp.float32)


class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        if int(mlp.width)!=1024 or int(mlp.depth)!=16 or int(budget)<2**41:
            return fnp.zeros((int(mlp.depth),int(mlp.width)),dtype=fnp.float32)
        return _predict_binary(mlp)
