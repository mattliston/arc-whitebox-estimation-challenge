# Exact per-pair signed cubic mode-1 Gram; cross-pair terms remain omitted.
_PCM_STATS = dict(joins=[])


def _pcm_metric_columns(J, coefficient, cohorts, start, layer):
    if layer == 7:
        _PCM_STATS['joins'].clear()
    if layer not in (7,10,13) or str(J.dtype) != 'float32':
        raise RuntimeError('PCM fixed join geometry')
    m,width = int(J.shape[0]),int(J.shape[1])
    Z = fnp.empty((m,width),dtype=fnp.float32)
    cursor = 0
    for absolute,count in cohorts:
        a = int(absolute)-int(start);count=int(count)
        if a != cursor or count <= 0 or count % 2 or a+count > width:
            raise RuntimeError('PCM intact ordered two-leg cohort')
        h=count//2
        hold=_E2Hold()
        plus=hold(coefficient[a:a+h]);minus=hold(coefficient[a+h:a+count])
        if not bool(fnp.all(minus == hold(-plus))):
            raise RuntimeError('PCM exact opposite signed coefficients')
        A=hold(J[:,a:a+h]);B=hold(J[:,a+h:a+count])
        A64=hold(A.astype(fnp.float64));B64=hold(B.astype(fnp.float64))
        x=hold(fnp.sum(hold(A64*A64),axis=0));y=hold(fnp.sum(hold(B64*B64),axis=0))
        dot=hold(fnp.sum(hold(A64*B64),axis=0))
        xy=hold(x*y)
        cross=hold(fnp.minimum(hold(dot*dot),xy)) # Cauchy roundoff clamp only
        detroot=hold(fnp.sqrt(hold(hold(fnp.maximum(hold(xy-cross),0.0))*hold(xy+cross))))
        xx=hold(x*x);yy=hold(y*y)
        den=hold(fnp.sqrt(hold(hold(xx+yy)+hold(2.0*detroot))))
        safe=hold(fnp.where(hold(den>0.0),den,1.0))
        inv=hold(hold(fnp.abs(hold(plus.astype(fnp.float64))))/safe)
        aa=hold(hold(hold(xx+detroot)*inv).astype(fnp.float32))
        ab=hold(hold(hold(-cross)*inv).astype(fnp.float32))
        bb=hold(hold(hold(yy+detroot)*inv).astype(fnp.float32))
        hold(fnp.add(hold(A*aa),hold(B*ab),out=Z[:,a:a+h]))
        hold(fnp.add(hold(A*ab),hold(B*bb),out=Z[:,a+h:a+count]))
        cursor += count
    if cursor != width or not bool(fnp.all(fnp.isfinite(Z))):
        raise RuntimeError('PCM finite complete metric columns')
    _PCM_STATS['joins'].append([int(layer),m,width,len(cohorts)])
    return Z
