"""RESEARCH ONLY: unpriced offline SVD; never a submission candidate.

Native-screen compatible module for a fixed necessary-condition test.
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import flopscope.numpy as fnp

BASE = Path(__file__).resolve().parent.parent / 'fable_wall_repair_v1/cand_c2_wallrepair_I95.py'
spec = importlib.util.spec_from_file_location('common_probe_base', BASE)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
original_bd = core._bd_update
layer = 0


def verify_plane_identity():
    rng = np.random.default_rng(3112)
    a, t, s = (rng.normal(size=8) for _ in range(3))
    old, new = (rng.normal(size=(8, 8)) for _ in range(2))
    np.fill_diagonal(old, 0)
    np.fill_diagonal(new, 0)
    born = np.array([1,1,0,1,0,1,1,0], dtype=bool)
    def plane(b):
        plus, minus = b+np.diag(t), b+np.diag(s)
        return (plus*plus*(a*born))@plus.T - (minus*minus*(a*born))@minus.T
    diff = new-old
    formula = (2*old*diff+diff*diff)*(a*(t-s)*born)[None,:]
    formula += (a*(t*t-s*s)*born)[:,None]*diff.T
    np.fill_diagonal(formula, 0)
    error = float(np.max(np.abs(plane(new)-plane(old)-formula)))
    assert error < 1e-12, error
    return error


PLANE_IDENTITY_ERROR = verify_plane_identity()


def projected_bd(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
    global layer
    ell = layer
    layer += 1
    base, delta = original_bd(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients)
    if ell < 4:
        return base, delta
    old = np.asarray(base, dtype=np.float32).astype(np.float64)
    active = np.asarray(alive, dtype=bool)
    born = np.asarray(births, dtype=bool)
    selected = np.flatnonzero(born)
    completed = old[:, selected].copy()
    completed[selected, np.arange(len(selected))] = (
        np.asarray(aux['gate'], dtype=np.float64)[selected]
        * np.asarray(aux['pre_v'], dtype=np.float64)[selected] * active[selected])
    u, singular, vt = np.linalg.svd(completed, full_matrices=False)
    projected = (u[:, :128] * singular[:128]) @ vt[:128]
    projected[~active] = 0
    projected[selected, np.arange(len(selected))] = 0
    new = old.copy()
    new[:, selected] = projected.astype(np.float32).astype(np.float64)
    diff = new-old
    aa = np.asarray(a, dtype=np.float32).astype(np.float64)
    tt = (np.asarray(t, dtype=np.float64)*active).astype(np.float32).astype(np.float64)
    ss = (np.asarray(s, dtype=np.float64)*active).astype(np.float32).astype(np.float64)
    weight = aa*(tt-ss)*born
    weight2 = aa*(tt*tt-ss*ss)*born
    correction = (2*old*diff+diff*diff)*weight[None, :] + weight2[:, None]*diff.T
    np.fill_diagonal(correction, 0)
    print(json.dumps(dict(common_projection_layer=ell, rank=128,
                          selected=len(selected),
                          offdiag_relative_change=float(np.linalg.norm(diff)/np.linalg.norm(old[:,selected])),
                          completed_energy_retained=float(np.sum(singular[:128]**2)/np.sum(singular**2)),
                          research_only_unpriced_svd=True)), flush=True)
    return fnp.asarray(new.astype(np.float32)), delta + fnp.asarray(correction)


core._bd_update = projected_bd


class Estimator(core.Estimator):
    def predict(self, mlp, budget):
        global layer
        layer = 0
        return super().predict(mlp, budget)
