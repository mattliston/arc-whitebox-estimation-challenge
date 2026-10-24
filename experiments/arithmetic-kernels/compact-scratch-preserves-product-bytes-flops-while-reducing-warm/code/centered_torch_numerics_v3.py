"""Centered native-CDF identities, with explicit runtime tensor clamp bounds."""
from pathlib import Path
from gaussian_cdf_quadrature_v1 import normal_cdf as quadrature_cdf
from tensor_clamp_min_v1 import tensor_clamp_min,rewrite_clamps

HERE = Path(__file__).resolve().parent
source = (HERE/'centered_torch_numerics_v1.py').read_text()
old = '    return torch.erfc(-alpha / math.sqrt(2)) / 2'
assert source.count(old)==1
source = source.replace(old,'    return quadrature_cdf(alpha)')
compiled,clamp_replacements = rewrite_clamps(source,__file__+':centered')
assert clamp_replacements==1
_env = dict(quadrature_cdf=quadrature_cdf,tensor_clamp_min=tensor_clamp_min)
exec(compiled,_env)
normal_cdf = _env['normal_cdf']
gaussian_central = _env['gaussian_central']
marginals = _env['marginals']
nonlinear = _env['nonlinear']
