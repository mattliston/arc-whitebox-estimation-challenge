"""Independent dynamic NumPy bank with supplied complete prefix/final payloads."""
from pathlib import Path
import numpy as np
from compact_fullpair_final_reference_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input, final_delta

HERE = Path(__file__).resolve().parent


def run(weights, P, Q):
    state = np.zeros((weights.shape[1], P['head'].shape[1]), np.float32)

    def hook(layer, mean, cov, k21, aux, weight):
        nonlocal state
        delta,state = gnn_step(layer,features(layer,mean,cov,k21,aux,weight),
                               aux['sigma'],weight,state,P)
        return delta

    def final_hook(layer,mean,aux,weight):
        h = final_input(final_scalar_features(layer,mean,aux,weight),weight,state,P)
        return final_delta(h,aux['sigma'],Q)

    result,_ = predict(weights,hook=hook,final_hook=final_hook,
                       gate_floor=.01,cov32=True,metric='covariance')
    return np.maximum(result,0)
