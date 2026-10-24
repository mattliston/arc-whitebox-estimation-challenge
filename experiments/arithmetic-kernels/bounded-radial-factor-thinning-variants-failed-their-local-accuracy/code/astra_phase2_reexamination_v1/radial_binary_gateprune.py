"""Accuracy emulation of a compact live-neuron path plus exact cold diagonals.

Means and univariate variances/skewness are retained for all neurons. Only
cross-neuron state involving nearly dead units is omitted. Nearly linear or
dead units may skip new signed-cube births; their current repeated moments
are restored in the one-step star. Full arrays here establish accuracy before
spending effort on variable-width billed products.
"""
import inspect
import numpy as np
import radial_binary_cubic as binary


def make_predict(gate_floor, birth_floor=None):
    if birth_floor is None:
        birth_floor = gate_floor
    source = inspect.getsource(binary.predict)
    old = '        g=aux["gate"];birth=aux["pdf"]/aux["sigma"]'
    new = '''        g=aux["gate"];birth=aux["pdf"]/aux["sigma"]
        alive = g >= gate_floor
        live = alive[:,None]*alive[None,:]
        births = (g >= birth_floor) & (g <= 1-birth_floor)
        selected = np.flatnonzero(births)
        cv = np.diag(C).copy()
        C = C*live
        np.fill_diagonal(C,cv)
        td = np.diag(target).copy()
        target = target*live
        np.fill_diagonal(target,td)'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '''        born=bp*bp*birth[None,:]+beta[:,None]*bp.T
        np.fill_diagonal(born,birth*(3*u*u+delta*delta)/4)
        cp_post=inherited+born
        p=np.concatenate([g.astype(np.float32)[:,None]*pp,p1.astype(np.float32),p2.astype(np.float32)],axis=1)
        lam=np.concatenate([lam,a.astype(np.float32),-a.astype(np.float32)])'''
    new = '''        born=bp*bp*(birth*births)[None,:]+(beta*births)[:,None]*bp.T
        np.fill_diagonal(born,birth*births*(3*u*u+delta*delta)/4)
        cp_post=(inherited+born)*live
        p=np.concatenate([g.astype(np.float32)[:,None]*pp,p1[:,selected].astype(np.float32),p2[:,selected].astype(np.float32)],axis=1)
        p=alive.astype(np.float32)[:,None]*p
        lam=np.concatenate([lam,a[selected].astype(np.float32),-a[selected].astype(np.float32)])'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    source = source.replace('star_rms=float(np.sqrt(np.mean(star*star)))',
                            'star_rms=float(np.sqrt(np.mean(star*star))),active=int(alive.sum()),births=int(births.sum()),columns=int(p.shape[1])')
    env = dict(vars(binary), gate_floor=gate_floor, birth_floor=birth_floor)
    exec(source, env)
    return env["predict"]
