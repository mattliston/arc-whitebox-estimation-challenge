"""MPS-supported Gaussian CDF, same mathematical shadow and v2 tapes."""
from pathlib import Path
here = Path(__file__).resolve().parent
source = (here/'current_transport_shadow_v1.py').read_text()
source = source.replace('rows_current_transport_shadow_v1','rows_current_transport_shadow_v2')
old = 'torch.erfc(-a/math.sqrt(2))/2'
assert source.count(old) == 1
source = source.replace(old,'(1+torch.erf(a/math.sqrt(2)))/2')
exec(compile(source,__file__,'exec'))
