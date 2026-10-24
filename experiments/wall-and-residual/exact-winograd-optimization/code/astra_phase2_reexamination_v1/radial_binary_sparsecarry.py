"""Sparse higher-order carrier with DENSE covariance and repeated closure.

Cold-neuron means, covariance cross entries, and repeated third moments are
not discarded. Only their multi-step cubic-factor rows are omitted; the
dense one-step star restores all current repeated entries. This separates
carrier sparsity from the more intrusive covariance pruning experiment.
"""
import inspect
import radial_binary_compact as base

PREDICT_SOURCE = inspect.getsource(base.predict)
PREDICT_SOURCE = PREDICT_SOURCE.replace('gate_floor=.001):', 'gate_floor=.01, birth_floor=.001):')
old = '''        star_pre = star_push(W[l], star)
        sd = np.diag(star_pre).copy()
        star_pre = star_pre*live
        np.fill_diagonal(star_pre, sd)'''
assert PREDICT_SOURCE.count(old) == 1
PREDICT_SOURCE = PREDICT_SOURCE.replace(old, '        star_pre = star_push(W[l], star)')
old = '''        selected, births = select(np.minimum(g, 1-g), gate_floor)
        cv = np.diag(C).copy()
        C = C*live
        np.fill_diagonal(C, cv)
        td = np.diag(target).copy()
        target = target*live
        np.fill_diagonal(target, td)'''
new = '''        selected, births = select(np.minimum(g, 1-g)*alive, birth_floor)'''
assert PREDICT_SOURCE.count(old) == 1
PREDICT_SOURCE = PREDICT_SOURCE.replace(old, new)
env = dict(vars(base))
exec(PREDICT_SOURCE, env)
predict = env["predict"]
