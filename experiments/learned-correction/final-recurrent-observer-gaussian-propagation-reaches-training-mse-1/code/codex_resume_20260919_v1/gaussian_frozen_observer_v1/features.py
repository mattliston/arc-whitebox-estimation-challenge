"""Current30-feature equations on explicitly Gaussian physical state."""

import flopscope.numpy as fnp

def features(l, m, C, K21, aux, W):
    n = len(m)
    s = aux['sigma']
    v = s * s
    rho = aux['pre_cov'] / fnp.outer(s, s)
    skplane = K21 / (v[:, None] * s[None, :])
    skplane_square = skplane * skplane
    postscale = fnp.sqrt(fnp.maximum(aux['post_v'], 1e-12))
    postrho = C / fnp.outer(postscale, postscale)
    postrho = fnp.clip(postrho, -1, 1)
    w = W.astype(fnp.float32)
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    ones = fnp.ones(n)
    vals = (ones * (2 * l / 15 - 1), ones * (l / 16), ones, aux['alpha'], fnp.log(s) - fnp.mean(fnp.log(s)), aux['gate'], aux['pdf'], aux['gaussian_mean'] / s, m / s, (m - aux['gaussian_mean']) / s, aux['pre_k3'] / (v * s), aux['post_v'] / v, aux['pre_k4'] / (v * v), fnp.mean(rho, axis=1), fnp.sqrt(fnp.mean(rho * rho, axis=1)), fnp.mean(fnp.abs(rho), axis=1), fnp.sum(w, axis=0) * 0.7071067811865475, fnp.sum(w2 - 0.001953125, axis=0) * 11.313708498984761, fnp.sum(w3, axis=0) * 93.47798314754834, fnp.sum(w4 - 1.1444091796875e-05, axis=0) * 836.0924988699916, aux['post_k3'] / (v * s), aux['post_k4'] / (v * v), fnp.sqrt(fnp.sum(skplane_square, axis=1)), fnp.sqrt(fnp.sum(skplane_square, axis=0)), fnp.sum(skplane, axis=1) / fnp.sqrt(n), fnp.sum(skplane, axis=0) / fnp.sqrt(n), rho @ aux['alpha'] / n, fnp.sqrt(fnp.mean(postrho * postrho, axis=1)), fnp.mean(postrho, axis=1), ones * fnp.mean(m / s))
    return fnp.stack(vals, axis=1).astype(fnp.float32)

def final_scalar_features(l,m,aux,W):
    n = len(m); s = aux["sigma"]; v = s*s
    w = W.astype(fnp.float32); w2 = w*w; w3 = w2*w; w4 = w2*w2
    ones = fnp.ones(n); zero = fnp.zeros(n)
    vals = (ones*(2*l/15-1),ones*(l/16),ones,aux["alpha"],fnp.log(s)-fnp.mean(fnp.log(s)),
            aux["gate"],aux["pdf"],aux["gaussian_mean"]/s,m/s,(m-aux["gaussian_mean"])/s,
            aux["pre_k3"]/(v*s),aux["post_v"]/v,aux["pre_k4"]/(v*v),zero,zero,zero,
            fnp.sum(w,axis=0)*0.7071067811865475,fnp.sum(w2-0.001953125,axis=0)*11.313708498984761,
            fnp.sum(w3,axis=0)*93.47798314754834,fnp.sum(w4-1.1444091796875e-5,axis=0)*836.0924988699916,
            aux["post_k3"]/(v*s),aux["post_k4"]/(v*v),zero,zero,zero,zero,zero,zero,zero,
            ones*fnp.mean(m/s))
    return fnp.stack(vals,axis=1).astype(fnp.float32)

def gaussian_features(layer,w,scalar,pre,post):
    s=scalar['s'];v=scalar['v'];a=scalar['a'];g=scalar['g'];p=scalar['pdf']
    normalized_mean=scalar['m']/s
    normalized_var=scalar['post_v']/v
    normalized_k3=scalar['k3']/(v*s)
    raw2=normalized_var+normalized_mean*normalized_mean
    raw3=normalized_k3+3*normalized_mean*raw2-2*normalized_mean**3
    raw4=(a**4+6*a*a+3)*g+(a**3+5*a)*p
    normalized_k4=(raw4-4*normalized_mean*raw3+6*normalized_mean**2*raw2
                   -3*normalized_mean**4-3*normalized_var**2)
    zero=fnp.zeros_like(s)
    aux=dict(sigma=s,alpha=a,pdf=p,gate=g,gaussian_mean=scalar['m'],
             pre_k3=zero,pre_k4=zero,post_v=scalar['post_v'],
             post_k3=scalar['k3'],post_k4=normalized_k4*v*v)
    if pre is None:
        return final_scalar_features(layer,scalar['m'],aux,w)
    aux['pre_cov']=pre
    return features(layer,scalar['m'],post,fnp.zeros_like(pre),aux,w)

