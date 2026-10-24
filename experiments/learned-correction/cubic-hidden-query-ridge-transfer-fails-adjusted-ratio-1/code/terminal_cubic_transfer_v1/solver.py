"""Fixed weighted-RMS ridge; no parameter or feature selection."""
def fit(np,design,target,weights):
    weights=weights/np.sum(weights)
    scale=np.maximum(np.sqrt(np.sum(weights[:,None]*design*design,axis=0)),1e-12)
    X=design/scale[None,:]
    gram=X.T@(weights[:,None]*X)+1e-3*np.eye(X.shape[1]);rhs=X.T@(weights*target)
    beta=np.linalg.solve(gram,rhs)
    residual=float(np.linalg.norm(gram@beta-rhs)/max(np.linalg.norm(rhs),1e-30));assert residual<=1e-9
    L=np.linalg.cholesky(gram);independent=np.linalg.solve(L.T,np.linalg.solve(L,rhs))
    relative=float(np.linalg.norm(beta-independent)/max(np.linalg.norm(beta),1e-30));assert relative<=1e-8
    return beta/scale,dict(scale=scale,scaled_beta=beta,residual=residual,cholesky_relative=relative)


def serve(fnp,parent,sigma,H,Z,beta):
    X=fnp.concatenate([fnp.ones((int(H.shape[0]),1),dtype=fnp.float64),H.astype(fnp.float64)],axis=1)
    A=sigma.astype(fnp.float64)[:,None]*X
    design=fnp.concatenate([A,Z],axis=1) if beta.shape[0]>X.shape[1] else A
    return fnp.maximum(parent.astype(fnp.float64)+design@beta,0).astype(fnp.float32)
