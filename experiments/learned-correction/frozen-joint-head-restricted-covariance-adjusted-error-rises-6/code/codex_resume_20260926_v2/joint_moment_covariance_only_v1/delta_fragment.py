def _jm_covariance_delta(l,mu,C,K,radial,W,H,aux):
    if int(H.shape[1]) != 64:
        raise RuntimeError('Frozen joint head requires H64')
    F,R,S,s=_jm_features(fnp,flops.stats.norm.cdf,l,mu,C,K,radial,W,H)
    a,b,c=_jm_predict_planes(fnp,F,R,S,_jm_parameters())
    return _jm_covariance_from_planes(l,a,b,c,aux)


def _jm_covariance_from_planes(l,a,b,c,aux):
    n=int(a.shape[0]);st=_e2_pair_state(n)
    pair=_E2_LAYER.get('pair')
    if pair is None or pair['aux'] is not aux:
        raise RuntimeError('Missing current pair coefficients')
    tc=_e2_pair_arena(n)['T']
    x21=(a*tc[0,0]+a.T*tc[1,0])/2
    x22=(c*tc[2,0])/4
    x31=(b*tc[3,0]+b.T*tc[4,0])/6
    dc=((x21+x22)+x31).astype(fnp.float64)*pair['ss']
    if l in _GN_COV_LAYERS:
        A1,A2,A3,A4,A5=fnp.unstack(st['FT'][1:6,0])
        rho=st['P16'][0];rho2=(rho*rho)/2
        kink=(fnp.abs(aux['alpha'])<1.5).astype(fnp.float64)
        dn=(a.astype(fnp.float64)*(_GN_KNOBS['k21n']*(1-kink))[:,None]).astype(fnp.float32)
        t21=(fnp.outer(A2,A1)+rho*fnp.outer(A3,A2))+rho2*fnp.outer(A4,A3)
        x=(dn*t21)/2
        dc=dc+(x+x.T).astype(fnp.float64)*pair['ss']
    fnp.fill_diagonal(dc,0.)
    return dc
