"""Research instrumentation: all array copies are counted FNP operations."""
import flopscope.numpy as fnp

def clone_fields(fields):
    return {key: fnp.copy(value) for key, value in fields.items()}

def factored_reference(fields, np):
    j, weight, omega = (fields[k].astype(np.float64) for k in ('J', 'weight', 'Omega'))
    y = j @ (weight[:, None] * (j.T @ omega))
    if 'Qp' in fields:
        q, c = (fields[k].astype(np.float64) for k in ('Qp', 'core'))
        y += q @ (c @ (q.T @ omega))
    return y

def physical(a, previous, np):
    result = np.zeros((1024, a.shape[1]), dtype=np.float64)
    result[previous] = a
    return result
