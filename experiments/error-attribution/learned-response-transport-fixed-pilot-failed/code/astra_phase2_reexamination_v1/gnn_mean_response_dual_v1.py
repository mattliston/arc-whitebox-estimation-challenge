"""Frozen GNN value and local raw-feature directional derivative."""
import numpy as np
from radial_gnn_numpy import silu


def step(l, x, dx, sigma, W, state, P, dtype=np.float32):
    if l < 1:
        return np.zeros_like(sigma), state*0, np.zeros_like(sigma)
    z = (x.astype(dtype)-P['centers'][l].astype(dtype))/P['scales'][l].astype(dtype)
    dz = dx.astype(dtype)/P['scales'][l].astype(dtype)
    dz *= (z > -8) & (z < 8)
    x = np.clip(z, -8, 8)
    g, dg = x[:, 5:6], dz[:, 5:6]
    # Previous state and weights are fixed under the virtual current bias.
    # These constants keep the exact original float32 aggregation even in
    # the float64 continuous-extension derivative certificate.
    src = state@P['msg_w'].T+P['msg_b']
    cm = int(P['msg_w'].shape[0])//3
    w = W.astype(np.float32)
    w2, w3 = w*w, w*w*w
    m1 = w.T@src[:, :cm]
    m2 = w2.T@src[:, cm:2*cm]
    m3 = w3.T@src[:, 2*cm:]
    e1 = w2.T@(src[:, :cm]**2)
    e2 = (w2*w2).T@(src[:, cm:2*cm]**2)
    e3 = (w3*w3).T@(src[:, 2*cm:]**2)
    msgs = np.concatenate([g*m1, g*m2, g*m3, np.log1p(np.maximum(e1, 0)),
                           np.log1p(np.maximum(e2, 0)), np.log1p(np.maximum(e3, 0))], axis=1)
    dmsgs = np.concatenate([dg*m1, dg*m2, dg*m3,
                            np.zeros_like(e1), np.zeros_like(e2), np.zeros_like(e3)], axis=1)
    dmsgs *= (msgs > -8) & (msgs < 8)
    h = np.concatenate([x, np.clip(msgs, -8, 8)], axis=1)
    dh = np.concatenate([dz, dmsgs], axis=1)
    for index in (0, 2):
        weight, bias = P[f'enc{index}_w'].astype(dtype), P[f'enc{index}_b'].astype(dtype)
        t = h@weight.T+bias
        dt = dh@weight.T
        h = silu(t)
        sigmoid = 1/(1+np.exp(-t))
        dh = dt*sigmoid*(1+t*(1-sigmoid))
    new_state = h@P['enc4_w'].astype(dtype).T+P['enc4_b'].astype(dtype)
    dstate = dh@P['enc4_w'].astype(dtype).T
    head = P['head'][l].astype(dtype)
    gain = 1+0.25*np.tanh(P['gain'][l])
    scale = sigma.astype(dtype)*P['rms'][l]
    delta = scale*(new_state@head+P['head_b'][l])*gain
    ddelta = scale*(dstate@head)*gain
    return delta, new_state, ddelta
