"""Isolate newborn pruning from the already-closed cold-state pruning test.

Existing factor support, covariance, star and trace-vector support still use
gate .01. Only NEW signed-cube births use the predeclared .05 threshold;
their omitted repeated entries are restored in the current star as usual.
This is not a new age/rank cap, and changes no carried factor after birth.
"""
import radial_compact_feed_fullpair_v1 as base

source=base.PREDICT_SOURCE
old='        selected, births = select(np.minimum(g, 1-g), gate_floor)'
new='        selected, births = select(np.minimum(g, 1-g), .05)'
assert source.count(old)==1
source=source.replace(old,new)
old='states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1]))'
new='states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1], birth05=True, clipped_selected_beta=int(np.count_nonzero(births & (np.abs(beta/safe_birth)>8*vv)))))'
assert source.count(old)==1
source=source.replace(old,new)
env=dict(base.env)
exec(source,env)
predict=env["predict"]
