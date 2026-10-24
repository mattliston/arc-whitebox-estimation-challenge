

# BEGIN FABLE-ENGINEER SELECT ROUNDING (E-8): selected counts rounded up to a multiple of _SELECT_ROUND (bank: 32) so that more
# product dimensions divide 2**6 (depth-6 pushes/squares); the extra units are real next-in-score units (predictions change).
def select(score, floor):
    count = min(len(score), (int(fnp.nonzero(score >= floor)[0].shape[0]) + _SELECT_ROUND - 1) // _SELECT_ROUND * _SELECT_ROUND)
    indices = fnp.argsort(-score, kind='stable')[:count]
    mask = fnp.zeros(len(score), dtype=fnp.bool_)
    fnp.put(mask, indices, True)
    return (indices, mask)
# END FABLE-ENGINEER SELECT ROUNDING
