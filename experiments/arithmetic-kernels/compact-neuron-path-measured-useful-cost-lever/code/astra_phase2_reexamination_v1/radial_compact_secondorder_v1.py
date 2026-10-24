"""Fixed second-order density completion on the source/full-pair host."""
import radial_binary_compact as base
from radial_compact_radialfeed_v1 import PREDICT_SOURCE
from radial_secondorder_v1 import make_nonlinear

env=dict(vars(base),nonlinear=make_nonlinear(6,2,True,True))
exec(PREDICT_SOURCE,env)
predict=env["predict"]
