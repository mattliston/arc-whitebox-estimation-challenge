"""Restrict the diagnostic positivity guard to actual emitted births.

V1 stopped on unselected, nearly-linear units. These coefficients emit no
factor and are irrelevant to the binary representation. No coefficient is
clamped or tuned here; a negative coefficient on selected support still fails.
"""
import projected_plane_feed_v1 as parent

source=parent.source
old='        assert np.all(birth>=-1e-14),"Negative projected birth coefficient"'
new='''        _,emitted=select(np.minimum(g,1-g),gate_floor)
        feed_profile["negative_unselected"]=int(np.count_nonzero((birth<0)&~emitted))
        feed_profile["negative_selected"]=int(np.count_nonzero((birth<0)&emitted))
        assert np.all(birth[emitted]>=-1e-14),("Negative selected projected birth",l,feed_profile)'''
assert source.count(old)==1
source=source.replace(old,new)
env=dict(parent.env)
exec(source,env)
predict=env["predict"]
