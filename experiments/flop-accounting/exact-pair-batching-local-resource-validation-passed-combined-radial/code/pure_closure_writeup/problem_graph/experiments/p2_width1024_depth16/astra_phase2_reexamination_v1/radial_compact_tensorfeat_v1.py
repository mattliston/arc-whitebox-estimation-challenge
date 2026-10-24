"""Unchanged compact host, with honest tensor-direction observables."""
import inspect
import numpy as np
import radial_binary_compact as base
from radial_features import features as old_features,FEATURE_NAMES as OLD_NAMES
from compact_tensor_directions_v1 import direction_features,DIRECTION_NAMES

FEATURE_NAMES=OLD_NAMES+DIRECTION_NAMES
source=inspect.getsource(base.predict)
old="        host = m.copy()"
new='        if hook is not None:\n            aux["tensor_directions"]=direction_features(mu,Cpre,pp,lam,W[l],star)\n'+old
assert source.count(old)==1
source=source.replace(old,new)
env=dict(vars(base),direction_features=direction_features)
exec(source,env)
predict=env["predict"]


def features(l,m,C,K21,aux,W):
    return np.concatenate([old_features(l,m,C,K21,aux,W),aux["tensor_directions"]],axis=1)
