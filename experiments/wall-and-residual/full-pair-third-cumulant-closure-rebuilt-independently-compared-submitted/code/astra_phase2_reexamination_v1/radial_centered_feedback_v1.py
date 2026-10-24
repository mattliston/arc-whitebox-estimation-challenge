"""Apply mean feedback as a deterministic translation of central moments."""
from radial_compact_radialfeed_v1 import PREDICT_SOURCE
import radial_binary_compact as base
from radial_mehler import make_nonlinear

source=PREDICT_SOURCE
old='            C = C-np.outer(m, dm)-np.outer(dm, m)-np.outer(dm, dm)'
assert source.count(old)==1
source=source.replace(old,'            # A deterministic translation leaves centered covariance unchanged.')
env=dict(vars(base));env['nonlinear']=make_nonlinear(6,2,True,True)
exec(source,env)
predict=env['predict']
