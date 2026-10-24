"""Inert FNP receiver-head fragment; appended only by the reviewed builder.

The estimator supplies fnp and the sole frozen _BDR_LITERAL. No array
arithmetic or third-party import occurs when this fragment is read.
"""

_BDR_PARAMETERS = None


def _bdr_parameters():
    global _BDR_PARAMETERS
    if _BDR_PARAMETERS is None:
        _BDR_PARAMETERS = fnp.asarray(_BDR_LITERAL, dtype=fnp.float32)
    return _BDR_PARAMETERS


def _bdr_modulation(H, R):
    """Four positive receiver-row factors from the original current H.

    Recompute the original NODE normalization in its exact FP32 order.
    Zero parameters are evaluated normally; there is no zero-value bypass.
    The return's first axis indexes the original four physical directions,
    and its second axis indexes receiving neurons, NOT birth columns.
    """
    n = int(H.shape[0])
    if H.shape != (n, 64) or R.shape != (64, 4):
        raise ValueError('Receiver-head shape mismatch')
    if any(str(x.dtype) != 'float32' for x in (H, R)):
        raise TypeError('Receiver-head operands must already be FP32')
    centered = H - fnp.mean(H, axis=0)
    scale = fnp.sqrt(fnp.mean(centered * centered, axis=0) + 1e-6)
    standardized = fnp.clip(centered / scale, -8, 8)
    return (1 + fnp.tanh(standardized @ R)).T
