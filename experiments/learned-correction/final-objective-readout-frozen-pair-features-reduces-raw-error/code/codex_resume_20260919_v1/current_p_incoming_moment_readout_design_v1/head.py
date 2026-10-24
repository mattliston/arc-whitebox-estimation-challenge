"""Causal features and one jointly supervised directed pair head."""
def features(xp,cdf,l,mu,C,K,radial,W,H):
    s=xp.sqrt(xp.maximum(xp.diag(C),1e-10));a=mu/s
    p=xp.exp(-.5*a*a)/2.5066282746310002;g=cdf(a)
    R=C/xp.outer(s,s);S=K/(s[:,None]**2*s[None,:]);n=len(s);logs=xp.log(s);ones=xp.ones(n)
    P=xp.stack((ones,ones*(l/15),a,g,p,a*p,(a*a-1)*p,logs-xp.mean(logs),xp.diag(S),ones*radial,
                xp.mean(R,axis=1),xp.sqrt(xp.mean(R*R,axis=1))),axis=1).astype(xp.float32)
    w=W.astype(xp.float32);h=H.astype(xp.float32)
    return xp.concatenate((P,w.T@h,(w*w).T@h),axis=1),R.astype(xp.float32),S.astype(xp.float32),s

def predict_planes(xp,F,R,S,t):
    F=xp.clip(F/t['feature_scale'],-8,8);R=R/t['rho_scale'];S=S/t['third_scale']
    u=F@t['left'];v=F@t['right'];blocks=[]
    for i in range(0,len(F),64):
        z=u[i:i+64,None,:]+v[None,:,:]+t['bias']
        z=z+R[i:i+64,:,None]*t['edge'][0]+S[i:i+64,:,None]*t['edge'][1]+S.T[i:i+64,:,None]*t['edge'][2]
        blocks.append((xp.tanh(z)@t['output'])*t['target_scale'])
    raw=xp.concatenate(blocks,axis=0);a=raw[:,:,0].copy();b=raw[:,:,1].copy();q=raw[:,:,2];c=.5*(q+q.T)
    for x in (a,b,c):xp.fill_diagonal(x,0.)
    return a,b,c

def correction(xp,cdf,l,mu,C,K,radial,W,H,t):
    F,R,S,s=features(xp,cdf,l,mu,C,K,radial,W,H);a,b,c=predict_planes(xp,F,R,S,t);v=xp.diag(C)
    return a.astype(xp.float64)*(v[:,None]*s[None,:]),b.astype(xp.float64),c.astype(xp.float64)*xp.outer(v,v)
