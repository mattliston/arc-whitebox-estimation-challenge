"""Separate cold-neuron pruning from near-linear birth pruning.

Versioned NumPy accuracy reference only. Optional scalar shadows retain the
next-layer variance and third-cumulant diagonal of omitted cross entries.
"""
import inspect
import radial_binary_compact as base


def make_predict(shadow=False):
    source = inspect.getsource(base.predict)
    source = source.replace("gate_floor=.001):", "gate_floor=.03, birth_floor=.01):")
    old = "selected, births = select(np.minimum(g, 1-g), gate_floor)"
    assert source.count(old) == 1
    source = source.replace(old, "selected, births = select(np.minimum(g, 1-g), birth_floor)")
    if shadow:
        edits = {
            "    means, states = [], []":
                "    means, states = [], []\n    cold_cov = np.zeros((n,n))\n    cold_star = np.zeros((n,n),np.float32)",
            "        vv = np.diag(Cpre)":
                "        shadow = w32.T @ cold_cov.astype(np.float32) @ w32\n        np.fill_diagonal(Cpre,np.diag(Cpre)+np.diag(shadow))\n        vv = np.diag(Cpre)",
            "        K21 = cp_pre+star_pre":
                "        K21 = cp_pre+star_pre\n        np.fill_diagonal(K21,np.diag(K21)+star_push(W[l],cold_star,diagonal_only=True))",
            "        cv = np.diag(C).copy()\n        C = C*live":
                "        cold_cov = C-C*live\n        np.fill_diagonal(cold_cov,0)\n        cold_star = (3*(target-target*live).T).astype(np.float32)\n        np.fill_diagonal(cold_star,0)\n        cv = np.diag(C).copy()\n        C = C*live",
        }
        for old, new in edits.items():
            assert source.count(old) == 1, old
            source = source.replace(old, new)
    env = dict(vars(base))
    exec(source, env)
    return env["predict"]


predict = make_predict(False)
predict_shadow = make_predict(True)
