"""Offline replay only; these helpers are never included in the candidate."""
import numpy as np


def replay(packet,head):
    delta=((packet['sigma'].astype(np.float32)*np.float32(packet['rms']))*
           (packet['hidden']@np.asarray(head[:64],dtype=np.float32)+np.float32(head[64])))
    delta=(delta*packet['effective_gain']).astype(np.float32)
    prediction=np.maximum(packet['host'].astype(np.float64)+delta.astype(np.float64),0).astype(np.float32)
    return prediction,delta
