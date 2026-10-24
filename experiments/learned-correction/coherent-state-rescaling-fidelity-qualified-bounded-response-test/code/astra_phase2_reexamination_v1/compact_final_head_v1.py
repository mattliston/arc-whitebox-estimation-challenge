"""Target-blind scalar-only final-layer inputs for a frozen compact prefix."""
import numpy as np


def final_input(x, W, state, P):
    x = np.clip((x.astype(np.float32)-P["centers"][15])/P["scales"][15], -8, 8)
    mask = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1], dtype=np.float32)
    x = x*mask[None,:]
    g = x[:,5:6]
    src = state@P["msg_w"].T+P["msg_b"]
    cm = int(P["msg_w"].shape[0])//3
    w = W.astype(np.float32); w2 = w*w; w3 = w2*w
    m1 = w.T@src[:,:cm]; m2 = w2.T@src[:,cm:2*cm]; m3 = w3.T@src[:,2*cm:]
    e1 = w2.T@(src[:,:cm]**2); e2 = (w2*w2).T@(src[:,cm:2*cm]**2); e3 = (w3*w3).T@(src[:,2*cm:]**2)
    msgs = np.concatenate([g*m1,g*m2,g*m3,np.log1p(np.maximum(e1,0)),np.log1p(np.maximum(e2,0)),np.log1p(np.maximum(e3,0))], axis=1)
    return np.concatenate([x,np.clip(msgs,-8,8)], axis=1)


def final_delta(h, sigma, Q):
    h = h@Q["w0"].T+Q["b0"]
    h = h/(1+np.exp(-h))
    h = h@Q["w2"].T+Q["b2"]
    h = h/(1+np.exp(-h))
    return sigma.astype(np.float32)*Q["rms"][0]*(h@Q["w4"][0]+Q["b4"][0])
