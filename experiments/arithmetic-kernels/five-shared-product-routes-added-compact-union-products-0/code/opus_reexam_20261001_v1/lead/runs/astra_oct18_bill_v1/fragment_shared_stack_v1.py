

# Public-shape exact L5/P40 extension. Existing SM12 runtime rules are immutable;
# only its setup plan inventory is extended to prebind these five full products.
_O18_SHARED_RULES = {(384, 960, 6144): (150217882, 8, ('signed_executable_0', 'o18shared_L5P40_4_10_32', 384, 960, 6144, (('parent', 0), ('parent', 0), ('parent', 0)))), (832, 384, 6144): (50222958, 11, ('signed_executable_0', 'o18shared_L5P40_9_4_32', 832, 384, 6144, (('parent', 0), ('parent', 0), ('parent', 0)))), (768, 6144, 384): (111852168, 26, ('signed_executable_1', 'o18shared_L5P40_8_32_4', 768, 6144, 384, (('parent', 0), ('parent', 0), ('parent', 0)))), (768, 384, 6144): (110625416, 8, ('signed_executable_0', 'o18shared_L5P40_8_4_32', 768, 384, 6144, (('parent', 0), ('parent', 0), ('parent', 0)))), (384, 832, 5568): (49460799, 0, ('signed_executable_0', 'o18shared_L5P40_4_9_29', 384, 832, 5568, (('parent', 0), ('parent', 0), ('parent', 0))))}
_O18_SHARED_FAMILIES = (('o18shared_L5P40_4_10_32', 5, 40, 4, 10, 32), ('o18shared_L5P40_4_9_29', 5, 40, 4, 9, 29), ('o18shared_L5P40_8_32_4', 5, 40, 8, 32, 4), ('o18shared_L5P40_8_4_32', 5, 40, 8, 4, 32), ('o18shared_L5P40_9_4_32', 5, 40, 9, 4, 32))
_O18_SHARED_PLANS = {(384, 960, 6144): ('signed_executable_0', 'o18shared_L5P40_4_10_32', 384, 960, 6144), (832, 384, 6144): ('signed_executable_0', 'o18shared_L5P40_9_4_32', 832, 384, 6144), (768, 6144, 384): ('signed_executable_1', 'o18shared_L5P40_8_32_4', 768, 6144, 384), (768, 384, 6144): ('signed_executable_0', 'o18shared_L5P40_8_4_32', 768, 384, 6144), (384, 832, 5568): ('signed_executable_0', 'o18shared_L5P40_4_9_29', 384, 832, 5568)}
assert not set(_O18_SHARED_RULES).intersection(_SM12_RULE)
for _o18_shared_family in _O18_SHARED_FAMILIES:
    _smx_register(_o18_shared_family)
_SM12_PLANS = dict(_SM12_PLANS, **{})
_SM12_PLANS.update(_O18_SHARED_PLANS)
_O18_SHARED_PARENT = _shared_old384_mm
_O18_SHARED_PARENT_PREDICT = Estimator.predict
_O18_SHARED_STATE = dict(left=64, accepted=0, rejected=0, selected=[])

def _shared_old384_mm(left, right):
    # Preserve every captured-reuse context and emergency route. The unchanged
    # parent owns existing SM12 keys, unsupported inputs and exhausted budgets.
    scope = _W5_PORT_SCOPE
    if (not _SMX_ON or not _AB_ON or not _WINOGRAD or _SCRATCH is None
            or _FAST_RESCUE_ON or _UBC_CONTEXT[0] is not None
            or _UBX_CTX[0] is not None or _ASTRA_CLUSTER_CTX[0] is not None
            or (scope is not None and scope.get('grant') is not None)
            or len(left.shape) != 2 or len(right.shape) != 2
            or left.dtype != 'float32' or right.dtype != 'float32'
            or int(left.shape[1]) != int(right.shape[0])):
        return _O18_SHARED_PARENT(left, right)
    key = (int(left.shape[0]), int(left.shape[1]), int(right.shape[1]))
    if key in _SM12_RULE:
        return _O18_SHARED_PARENT(left, right)
    option = _O18_SHARED_RULES.get(key)
    if option is not None:
        gain, cost, plan = option
        if cost <= _O18_SHARED_STATE['left']:
            _O18_SHARED_STATE['left'] -= cost
            _O18_SHARED_STATE['accepted'] += 1
            _O18_SHARED_STATE['selected'].append((key, cost))
            return _peel_product(left, right, plan)
        _O18_SHARED_STATE['rejected'] += 1
    return _O18_SHARED_PARENT(left, right)

def _o18_shared_predict(self, mlp, budget):
    _O18_SHARED_STATE.update(left=64, accepted=0, rejected=0, selected=[])
    return _O18_SHARED_PARENT_PREDICT(self, mlp, budget)

Estimator.predict = _o18_shared_predict
