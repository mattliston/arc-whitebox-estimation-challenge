# Research only: carry the Gram selection core at the current coefficient scale.
# The loop multiplies every inherited lambda by s_carry after commit. The core
# approximates F diag(lambda**2) F.T, so its matching scalar is s_carry**2.
# This changes later basis choices; it is not a prediction-preserving refactor.
_CRS_OLD_COMMIT = _shared_old384_commit
_CRS_STATS = {'rescaled': 0}


def _shared_old384_commit(layer, gate, alive, selected, state):
    _CRS_OLD_COMMIT(layer, gate, alive, selected, state)
    if _S19_LAM is not None and state.get('core') is not None:
        state['core'] = state['core'] * (_S19_LAM[0] ** 2)
        _CRS_STATS['rescaled'] += 1
