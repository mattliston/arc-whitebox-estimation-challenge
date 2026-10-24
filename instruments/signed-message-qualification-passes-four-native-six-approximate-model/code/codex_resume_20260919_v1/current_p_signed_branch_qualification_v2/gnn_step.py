def gnn_step(l,x,sigma,W,state,P):
    if l<1:
        return fnp.zeros_like(sigma),state*0
    x = fnp.clip((x.astype(fnp.float32)-P["centers"][l])/P["scales"][l],-8,8)
    if l == 15:
        x = x * fnp.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=fnp.float32)[None, :]
    g = x[:,5:6]
    src = state@P["msg_w"].T+P["msg_b"]
    cm = int(P["msg_w"].shape[0])//3
    w = W.astype(fnp.float32); w2=w*w; w3=w2*w
    m1 = w.T@src[:,:cm]; m2=w2.T@src[:,cm:2*cm]; m3=w3.T@src[:,2*cm:]
    e1=w2.T@(src[:,:cm]**2); e2=(w2*w2).T@(src[:,cm:2*cm]**2); e3=(w3*w3).T@(src[:,2*cm:]**2)
    msgs=fnp.concatenate([g*m1,g*m2,g*m3,fnp.log1p(fnp.maximum(e1,0)),fnp.log1p(fnp.maximum(e2,0)),fnp.log1p(fnp.maximum(e3,0))],axis=1)
    h=fnp.concatenate([x,fnp.clip(msgs,-8,8)],axis=1)
    signed=fnp.clip(fnp.concatenate([m1,m2,m3],axis=1),-8,8)
    pre=h@P["enc0_w"].T+P["enc0_b"]
    h=silu(pre+signed@P["ungated_w"].T)
    h=silu(h@P["enc2_w"].T+P["enc2_b"])
    state=h@P["enc4_w"].T+P["enc4_b"]
    delta=(sigma.astype(fnp.float32)*P["rms"][l])*(state@P["head"][l]+P["head_b"][l])*(1+0.25*fnp.tanh(P["gain"][l:l+1]))
    return delta,state
