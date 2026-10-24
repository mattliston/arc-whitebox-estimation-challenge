"""Consistent higher pair map on the new fourth-to-third interaction host.

This composes the previously measured full first-order pair map with the
newly supplied trace tensor. It tests the consumer on the changed state;
the old pair-map-only mixed result is not silently treated as a success.
"""
import radial_binary_compact as base
from radial_compact_radialfeed_v1 import PREDICT_SOURCE
from radial_mehler import make_nonlinear

env=dict(vars(base))
env["nonlinear"]=make_nonlinear(6,2,True,True)
exec(PREDICT_SOURCE,env)
predict=env["predict"]
