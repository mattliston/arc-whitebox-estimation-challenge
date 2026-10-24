"""Actual same-size CP birth update; no target data or new readout."""
import numpy as np
import radial_compact_feed_fullpair_v1 as parent
from positive_path_resummation_v1 import factors,positive_factor


def resummed_columns(C,K21,K22,aux,radial,enabled):
    g=aux['gate'];d=aux['pdf']/aux['sigma'];eta=-aux['alpha']/aux['sigma']
    B,E=factors(C,g,d,eta,K21,K22,radial)
    if not enabled:return B,dict(negative_relative=0.,shift_relative=0.)
    Z,s=positive_factor(B,E)
    negative=np.linalg.norm(s['negative']*s['norm2'])
    positive=np.linalg.norm(s['positive']*s['norm2'])
    return Z,dict(negative_relative=float(negative/max(positive,1e-30)),
                  shift_relative=float(np.linalg.norm(Z-B)/max(np.linalg.norm(B),1e-30)))


def compose(source):
    old='gate_floor=.001):';assert source.count(old)==1
    source=source.replace(old,'gate_floor=.001, path_mode=True):')
    old='        bp = g[:, None]*Cpre';assert source.count(old)==1
    source=source.replace(old,'        bp,path_stats = resummed_columns(Cpre,K21,k22,aux,radial,path_mode)')
    old='states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1]))'
    assert source.count(old)==1
    source=source.replace(old,'states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1],**path_stats))')
    return source


source=compose(parent.PREDICT_SOURCE)
env=dict(parent.env,resummed_columns=resummed_columns)
exec(compile(source,__file__,'exec'),env)
predict=env['predict']
