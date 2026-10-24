"""Inert positive-diagonal normalization; all data operations remain paid."""


def _unitnorm_weights(weights):
    depth = len(weights)
    if depth < 2 or len(weights[0].shape) != 2:
        raise RuntimeError('Unitnorm weight-list contract')
    n = int(weights[0].shape[0])
    previous = fnp.ones(n, dtype=fnp.float64)
    transformed, scales = [], []
    for layer in range(depth):
        if tuple(weights[layer].shape) != (n, n):
            raise RuntimeError('Unitnorm square-weight contract')
        weight = fnp.asarray(weights[layer], dtype=fnp.float64)
        if not bool(fnp.all(fnp.isfinite(weight))):
            raise RuntimeError('Unitnorm finite-weight contract')
        value = weight / previous[:, None]
        if layer < depth - 1:
            energy = fnp.sum(value * value, axis=0)
            nonzero = energy > 0
            denominator = fnp.where(nonzero, energy, 1.0)
            scale = fnp.where(nonzero, fnp.sqrt(2.0 / denominator), 1.0)
            balanced = (value * scale[None, :]).astype(fnp.float32)
        else:
            scale = fnp.ones(n, dtype=fnp.float64)
            balanced = value.astype(fnp.float32)
        if (not bool(fnp.all(fnp.isfinite(scale)))
                or not bool(fnp.all(scale > 0))
                or not bool(fnp.all(fnp.isfinite(balanced)))):
            raise RuntimeError('Unitnorm transformed domain')
        transformed.append(balanced)
        scales.append(scale)
        previous = scale
    return transformed, fnp.stack(scales, axis=0)


class _UnitnormProxy:
    def __init__(self, mlp, weights):
        self.width = int(mlp.width)
        self.depth = int(mlp.depth)
        self.weights = weights


def _unitnorm_predict(mlp):
    weights, scales = _unitnorm_weights(mlp.weights)
    proxy = _UnitnormProxy(mlp, weights)
    result = _predict_learned_compact(proxy)
    return (result.astype(fnp.float64) / scales).astype(fnp.float32)
