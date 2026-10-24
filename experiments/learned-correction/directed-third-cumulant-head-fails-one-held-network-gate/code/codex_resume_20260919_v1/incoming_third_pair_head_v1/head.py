"""One causal incoming-third head. The namespace xp may be NumPy or FNP."""
def features(xp,cdf,l,mu,C,K,radial,W,H):
    s=xp.sqrt(xp.maximum(xp.diag(C),1e-10));a=mu/s
    p=xp.exp(-.5*a*a)/2.5066282746310002;g=cdf(a)
    R=C/xp.outer(s,s);S=K/(s[:,None]**2*s[None,:]);n=len(s)
    logs=xp.log(s);ones=xp.ones(n)
    P=xp.stack((ones,ones*(l/15),a,g,p,a*p,(a*a-1)*p,logs-xp.mean(logs),
                xp.diag(S),ones*radial,xp.mean(R,axis=1),xp.sqrt(xp.mean(R*R,axis=1))),axis=1).astype(xp.float32)
    w=W.astype(xp.float32);h=H.astype(xp.float32)
    F=xp.concatenate((P,w.T@h,(w*w).T@h),axis=1)
    return F,R.astype(xp.float32),S.astype(xp.float32),s

def predict_s(xp,F,R,S,t):
    F=xp.clip(F/t['feature_scale'],-8,8);R=R/t['rho_scale'];S=S/t['third_scale']
    u=F@t['left'];v=F@t['right'];blocks=[];n=len(F)
    for i in range(0,n,64):
        z=u[i:i+64,None,:]+v[None,:,:]+t['bias']
        z=z+R[i:i+64,:,None]*t['edge'][0]+S[i:i+64,:,None]*t['edge'][1]+S.T[i:i+64,:,None]*t['edge'][2]
        blocks.append((xp.tanh(z)@t['output'])*t['target_scale'])
    result=xp.concatenate(blocks,axis=0);xp.fill_diagonal(result,0.)
    return result

def correction(xp,cdf,l,mu,C,K,radial,W,H,t):
    F,R,S,s=features(xp,cdf,l,mu,C,K,radial,W,H)
    sf=s.astype(xp.float32)
    return (predict_s(xp,F,R,S,t)*(sf[:,None]**2*sf[None,:])).astype(xp.float64)
