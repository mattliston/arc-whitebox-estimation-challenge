_FULL_CUBIC_SUPPORT = True

def _cubic_support_select(score, floor):
    if not _FULL_CUBIC_SUPPORT:
        return select(score, floor)
    indices = fnp.argsort(-score, kind='stable')
    mask = fnp.ones(len(score), dtype=fnp.bool_)
    return indices, mask

