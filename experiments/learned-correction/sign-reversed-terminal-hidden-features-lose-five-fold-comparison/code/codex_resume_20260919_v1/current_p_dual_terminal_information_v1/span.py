"""Fixed cross-fit residual span with train-only weighted orthogonalization."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent.parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
RIDGE = 1e-4


def rms(x, w):
    return np.sqrt(np.sum(x*x*w[:, None], axis=0)/np.sum(w))


def ridge(x, y, w):
    gram = x.T @ (x*w[:, None])
    gram = (gram + gram.T)*.5
    gram.flat[::len(gram)+1] += RIDGE*np.sum(w)
    chol = np.linalg.cholesky(gram)
    return np.linalg.solve(chol.T, np.linalg.solve(chol, x.T @ (w*y)))


def fit(x, z, y, w):
    sx, sz = np.maximum(rms(x, w), 1e-12), np.maximum(rms(z, w), 1e-12)
    xn, zn = x/sx, z/sz
    beta = ridge(xn, y, w)
    root_w = np.sqrt(w)
    u, singular, vt = np.linalg.svd(root_w[:, None]*xn, full_matrices=False)
    retain = singular > 1e-7*singular[0]
    assert retain.any()
    projection = (vt[retain].T/singular[retain]) @ (u[:, retain].T @ (root_w[:, None]*zn))
    residual = zn - xn @ projection
    sr = rms(residual, w)
    keep = sr > 1e-6
    en = residual[:, keep]/sr[keep]
    gamma = ridge(en, y-xn@beta, w) if keep.any() else np.empty(0)
    orthogonality = float(np.max(np.abs(xn.T @ (en*w[:, None])))/np.sum(w)) if keep.any() else 0.
    assert orthogonality < 1e-5, orthogonality
    return dict(sx=sx, sz=sz, beta=beta, projection=projection, sr=sr,
                 keep=keep, gamma=gamma, rank=int(retain.sum()), singular=singular,
                 orthogonality=orthogonality)


def predict(model, x, z):
    xn = x/model['sx']
    control = xn @ model['beta']
    residual = z/model['sz'] - xn @ model['projection']
    extension = (residual[:, model['keep']]/model['sr'][model['keep']]) @ model['gamma']
    return control, control+extension


def certificate():
    # Duplicate features cannot change the control prediction, including off-train.
    t = np.linspace(-1, 1, 40)
    x = np.stack([np.ones_like(t), t, t*t], axis=1)
    z = x @ np.array([[2., -1.], [.25, 3.], [-.7, 1.2]])
    y, w = np.sin(2*t), 1.2+.2*t
    duplicate = fit(x, z, y, w)
    assert not duplicate['keep'].any()
    a, b = predict(duplicate, x, z)
    assert np.array_equal(a, b)
    # A direct augmented normal-equation solve independently checks the two-stage fit.
    z = np.stack([np.sin(3*t), np.cos(2*t), t**3], axis=1)
    model = fit(x, z, y, w)
    xn = x/model['sx']
    en = (z/model['sz'] - xn@model['projection'])[:, model['keep']]/model['sr'][model['keep']]
    design = np.concatenate([xn, en], axis=1)
    gram = design.T @ (design*w[:, None]) + RIDGE*np.sum(w)*np.eye(design.shape[1])
    joint = np.linalg.solve(gram, design.T @ (w*y))
    _, staged = predict(model, x, z)
    error = float(np.max(np.abs(staged - design@joint)))
    assert error < 1e-10, error
    return dict(duplicate_columns_leave_control_exact=True, augmented_solve_max_abs=error,
                 training_orthogonality=model['orthogonality'])

