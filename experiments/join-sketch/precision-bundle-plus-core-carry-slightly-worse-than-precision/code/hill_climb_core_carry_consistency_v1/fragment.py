# One fixed nominal squared-weight carry update; no tensor/read change.
_CORE_CARRY_PARENT_COMMIT = _shared_old384_commit


def _shared_old384_commit(layer, gate, alive, selected, state):
    result = _CORE_CARRY_PARENT_COMMIT(layer, gate, alive, selected, state)
    core = state['core']
    if core is not None and _S19_LAM is not None:
        state['core'] = core * (_S19_LAM[0] * _S19_LAM[0])
    return result
