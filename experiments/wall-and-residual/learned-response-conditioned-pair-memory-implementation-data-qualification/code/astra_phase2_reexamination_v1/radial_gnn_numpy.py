"""Mean-head-only GNN inference, matching the inherited honest trainer."""
import numpy as np

def silu(x):
    return x/(1+np.exp(-x))

def gnn_step(l,x,sigma,W,state,P):
    if l<1:
        return np.zeros_like(sigma),state*0
    x = np.clip((x.astype(np.float32)-P["centers"][l])/P["scales"][l],-8,8)
    g = x[:,5:6]
    src = state@P["msg_w"].T+P["msg_b"]
    cm = int(P["msg_w"].shape[0])//3
    w = W.astype(np.float32); w2=w*w; w3=w2*w
    m1 = w.T@src[:,:cm]; m2=w2.T@src[:,cm:2*cm]; m3=w3.T@src[:,2*cm:]
    e1=w2.T@(src[:,:cm]**2); e2=(w2*w2).T@(src[:,cm:2*cm]**2); e3=(w3*w3).T@(src[:,2*cm:]**2)
    msgs=np.concatenate([g*m1,g*m2,g*m3,np.log1p(np.maximum(e1,0)),np.log1p(np.maximum(e2,0)),np.log1p(np.maximum(e3,0))],axis=1)
    h=np.concatenate([x,np.clip(msgs,-8,8)],axis=1)
    h=silu(h@P["enc0_w"].T+P["enc0_b"])
    h=silu(h@P["enc2_w"].T+P["enc2_b"])
    state=h@P["enc4_w"].T+P["enc4_b"]
    delta=(sigma.astype(np.float32)*P["rms"][l])*(state@P["head"][l]+P["head_b"][l])*(1+0.25*np.tanh(P["gain"][l]))
    return delta,state
