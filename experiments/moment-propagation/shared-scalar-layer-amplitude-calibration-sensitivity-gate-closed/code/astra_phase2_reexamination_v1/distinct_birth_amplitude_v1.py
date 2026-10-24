"""Layer-only amplitude action, exact current repeated-plane restoration."""
import radial_compact_feed_fullpair_v1 as parent

source = parent.PREDICT_SOURCE
old = 'collect_states=False, gate_floor=.001):'
assert source.count(old) == 1
source = source.replace(old, 'collect_states=False, gate_floor=.001, birth_amplitudes=None):')
old = '        cp_post = (inherited+born)*live'
assert source.count(old) == 1
source = source.replace(old, '        amplitude = 1.0 if birth_amplitudes is None else birth_amplitudes[l]\n'
                         '        born = born*amplitude\n'
                         '        a = a*amplitude\n'+old)
env = dict(parent.env)
exec(compile(source, __file__, 'exec'), env)
predict = env['predict']
