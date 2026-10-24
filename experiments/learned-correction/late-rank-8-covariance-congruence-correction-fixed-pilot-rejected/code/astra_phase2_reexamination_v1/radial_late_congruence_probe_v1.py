"""Research-only action hook on the unchanged current full carrier."""
import radial_compact_feed_fullpair_v1 as parent

source = parent.PREDICT_SOURCE
old = 'collect_states=False, gate_floor=.001):'
assert source.count(old) == 1
source = source.replace(old, 'collect_states=False, gate_floor=.001, cov_action=None):')
old = '        np.fill_diagonal(C, cv)'
assert source.count(old) == 1
source = source.replace(old, old+'\n        if cov_action is not None:\n            C = cov_action(l, m, C, aux)')
env = dict(parent.env)
exec(compile(source, __file__, 'exec'), env)
predict = env['predict']
