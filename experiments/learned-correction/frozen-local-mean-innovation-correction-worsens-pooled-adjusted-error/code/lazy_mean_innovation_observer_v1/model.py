"""Shared FP32 source MLP; xp is NumPy or legal FlopScope NumPy."""
def inputs(features,hidden,layer,mu,sd,xp):
    assert 8<=layer<=15 and features.shape[1]==30 and hidden.shape[1]==64
    x=(xp.concatenate([features,hidden],axis=1)-mu)/sd
    one=xp.asarray([1.0 if j==layer-8 else 0.0 for j in range(8)],dtype=xp.float32)
    embedding=xp.ones((int(x.shape[0]),1),dtype=xp.float32)*one[None,:]
    return xp.concatenate([x,embedding],axis=1)
def forward(x,p,xp):
    return xp.maximum(x@p['W1']+p['b1'],0)@p['W2']+p['b2']
