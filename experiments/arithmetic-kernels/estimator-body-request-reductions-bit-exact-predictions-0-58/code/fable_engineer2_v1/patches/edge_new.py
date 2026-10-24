_E2_EDGE_STATE = {}   # E2: persistent float32 R/S planes (+ broadcast views) and the edge-weight rows of the message edge source


def _e2_edge_consts(t, n):
    k = _E2_EDGE_STATE.get(n)
    if k is not None and k['t'] is t:
        return k
    R = fnp.empty((n, n), dtype=fnp.float32)
    S = fnp.empty((n, n), dtype=fnp.float32)
    e0, e1, e2 = fnp.unstack(t['edge'])
    k = dict(t=t, R=R, S=S, R3=R[:, :, None], S3=S[:, :, None], ST3=S.T[:, :, None], e0=e0, e1=e1, e2=e2)
    _E2_EDGE_STATE[n] = k
    return k


def _message_edge_source(layer,m,K21,aux,w,state,P,context):
    global _EDGE_PARAMETERS
    if not 8<=layer<=14:
        raise ValueError('Frozen covariance edge layer')
    if _EDGE_PARAMETERS is None:
        _EDGE_PARAMETERS={k:fnp.asarray(v,dtype=fnp.float32) for k,v in _EDGE_LITERAL.items()}
    t=_EDGE_PARAMETERS
    n=int(m.shape[0])
    k=_e2_edge_consts(t,n)
    a=aux['alpha'];p=aux['pdf'];g=aux['gate'];s=aux['sigma']
    sk=aux['pre_k3']/s**3;ku=aux['pre_k4']/s**4;ms=m/s
    F=fnp.stack((g,p,a*p,(a*a-1)*p,ms,a*g,sk*g,sk*p,sk*a*p,sk*ms,
                 ku*g,ku*p,ku*a*p,ku*ms),axis=1,dtype=fnp.float32)
    R=k['R'];S=k['S']
    fnp.divide(aux['pre_cov'],fnp.outer(s,s),out=R)
    fnp.divide(K21,fnp.outer(s**2,s),out=S)
    F=F/t['feature_scale'];fnp.divide(R,t['rho_scale'],out=R);fnp.divide(S,t['third_scale'],out=S)
    # Draft: the previous layer's state is transported into current coordinates.
    # Both new48x16 matrices must begin at positive zero. All old weights freeze.
    cm = int(P['msg_w'].shape[0])//3
    if cm != 16 or context['left'].shape != (48,16) or context['right'].shape != (48,16):
        raise ValueError('Message covariance draft shape mismatch')
    gk=_e2_gnn_consts(P,n)
    fnp.clip(_e2_messages(gk,w,state),-8,8,out=gk['MSG3'])
    messages=gk['MSG']
    left=F@t['left']+messages@context['left']
    right=F@t['right']+messages@context['right']
    h=left[:,None,:]+right+t['bias']
    fnp.add(h,k['R3']*k['e0'],out=h)
    fnp.add(h,k['S3']*k['e1'],out=h)
    fnp.add(h,k['ST3']*k['e2'],out=h)
    sf=s.astype(fnp.float32);pf=p.astype(fnp.float32);gf=g.astype(fnp.float32)
    fnp.tanh(h,out=h)
    raw=(h@t['output'])*pf[:,None]*gf
    return (0.5*(raw+raw.T)*fnp.outer(sf,sf)*t['cov_factor']).astype(fnp.float64)
