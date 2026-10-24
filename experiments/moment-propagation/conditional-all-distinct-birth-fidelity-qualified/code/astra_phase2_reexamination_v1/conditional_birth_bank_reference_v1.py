"""Conditional birth prefix composed with the exact existing scalar endpoint."""
import conditional_distinct_birth_v1 as birth
import compact_fullpair_final_reference_v1 as final

source = birth.PREDICT_SOURCE
old = 'gate_floor=.001, birth_hook=None):'
assert source.count(old) == 1
source = source.replace(old,'gate_floor=.01, birth_hook=None, final_hook=None):')
marker = '        Cpre = (w32.T @ C.astype(np.float32) @ w32).astype(np.float64) if cov32 else w.T @ C @ w'
start = final.source.index('        if l == depth-1:\n            pushed = C.astype(np.float32)@w32')
end = final.source.index(marker,start)
block = final.source[start:end]
assert source.count(marker) == 1
source = source.replace(marker,block+marker)
env = dict(birth.env,nonlinear_scalar=final.env['nonlinear_scalar'])
exec(source,env)
predict = env['predict']
