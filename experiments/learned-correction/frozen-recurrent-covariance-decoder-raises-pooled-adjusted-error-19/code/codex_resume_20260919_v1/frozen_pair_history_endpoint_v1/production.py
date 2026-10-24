# Appended to the exact parent; all numeric work occurs during prediction.
_HISTORY_ENABLED=True
_HISTORY_PARAMETERS=None

def _history_parameters():
    global _HISTORY_PARAMETERS
    if _HISTORY_PARAMETERS is None:
        _HISTORY_PARAMETERS={'P':_unpack_static_fp32(_HISTORY_P0_LITERAL),
                             'Q':_unpack_static_fp32(_HISTORY_MODEL_LITERAL)}
    return _HISTORY_PARAMETERS

def _history_state_step(layer,m,C,K21,aux,w,state):
    p=_history_parameters()['P']
    if state is None:
        state=fnp.zeros((len(m),64),dtype=fnp.float32)
    x=features(layer,m,C,K21,aux,w)
    _,state=gnn_step(layer,x,aux['sigma'],w,state,p)
    return state

def _history_covariance(state,aux):
    p=_history_parameters()['Q']
    s=aux['sigma'];sf=s.astype(fnp.float32)
    F=fnp.concatenate((state,fnp.ones((int(state.shape[0]),1),dtype=fnp.float32)),axis=1)
    F=(F/p['feature_scale'])*(sf/p['sigma_scale'])[:,None]
    R=(aux['pre_cov']/fnp.outer(s,s)).astype(fnp.float32)
    value=(F@p['theta'][0])@F.T
    value=value+((F@p['theta'][1])@F.T)*R
    value=value+((F@p['theta'][2])@F.T)*(R*R)
    return (value*p['cov_scale']).astype(fnp.float64)
