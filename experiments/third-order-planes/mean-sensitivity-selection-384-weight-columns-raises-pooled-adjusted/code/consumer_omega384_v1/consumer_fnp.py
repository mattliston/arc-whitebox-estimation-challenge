"""Local third-cumulant sensitivity selects the fixed Omega column count."""


def _cs384_indices(mu, variance):
    if tuple(mu.shape) != (1024,) or tuple(variance.shape) != (1024,):
        raise RuntimeError('Consumer Omega requires1024 means and variances')
    v = fnp.maximum(variance, 1e-10)
    alpha = mu / fnp.sqrt(v)
    score = fnp.abs(alpha) * flops.stats.norm.pdf(alpha) / v
    return fnp.argsort(-score, kind='stable')[:384]


def _cs384_omega(w32, previous, indices):
    if tuple(w32.shape) != (1024, 1024) or tuple(indices.shape) != (384,):
        raise RuntimeError('Consumer Omega fixed matrix and column shapes')
    return w32[previous[:, None], indices[None, :]]
