"""Centered v1 identities with a native quadrature CDF in private globals."""
import hashlib
import json
from pathlib import Path
from gaussian_cdf_quadrature_v1 import normal_cdf as quadrature_cdf

HERE = Path(__file__).resolve().parent
source = (HERE/'centered_torch_numerics_v1.py').read_text()
parent = json.loads((HERE/'centered_torch_numerics_cpu_small_v1.json').read_text())
assert parent['passed'] and hashlib.sha256(source.encode()).hexdigest()==parent['source_sha256']
old = '    return torch.erfc(-alpha / math.sqrt(2)) / 2'
assert source.count(old)==1
source = source.replace(old,'    return quadrature_cdf(alpha)')
_env = dict(quadrature_cdf=quadrature_cdf)
exec(compile(source,__file__+':centered','exec'),_env)
normal_cdf = _env['normal_cdf']
gaussian_central = _env['gaussian_central']
marginals = _env['marginals']
nonlinear = _env['nonlinear']
