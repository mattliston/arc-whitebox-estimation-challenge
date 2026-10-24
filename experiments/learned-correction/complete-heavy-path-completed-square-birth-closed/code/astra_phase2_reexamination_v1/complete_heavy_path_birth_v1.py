"""All same-order two-edge path legs, completed square; NumPy research."""
import numpy as np
import radial_compact_feed_fullpair_v1 as parent


def shift_columns(C, T, K22, aux, radial):
    g = aux['gate']
    d = aux['pdf']/aux['sigma']
    eta = -aux['alpha']/aux['sigma']
    delta = (.5*g[:, None]*T.T*eta[None, :]
             +.5*d[:, None]*T
             +d[:, None]*eta[None, :]*(K22/4+C*C/2))
    np.fill_diagonal(delta, 0)
    delta /= 1+radial/3
    return delta


source = parent.PREDICT_SOURCE
old = '        bp = g[:, None]*Cpre'
assert source.count(old) == 1
source = source.replace(old, old+'\n        path_shift = shift_columns(Cpre,K21,k22,aux,radial)\n'
                         '        path_shift_relative = float(np.linalg.norm(path_shift)/max(np.linalg.norm(bp),1e-30))\n'
                         '        bp = bp+path_shift')
old = 'states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1]))'
assert source.count(old) == 1
source = source.replace(old, 'states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1],path_shift_relative=path_shift_relative))')
env = dict(parent.env, shift_columns=shift_columns)
exec(compile(source, __file__, 'exec'), env)
predict = env['predict']
