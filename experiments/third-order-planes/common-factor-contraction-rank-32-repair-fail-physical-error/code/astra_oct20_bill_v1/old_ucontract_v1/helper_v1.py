"""Fixed L10 CP-pair u-residual coreset with the full row-mean retained.

The full v leg, full carried bank, diagonal and NP forms are untouched.
Selection is frozen for the immutable original factor owner, so existing B
transform caching remains valid under a separate cache namespace.
"""
_O21_UPAIR_CACHE = {}
_O21_UPAIR_STATS = dict(selected_generations=0, calls=0, original_pairs=0, retained_pairs=0)


def _o21_upair_inner(Su, Fu, uv):
    run = _UBC_RUN[0]
    layer = _UVP_CUR[0]
    if (run is None or run.get('uv') is not uv or run.get('owner') is None
            or layer not in (10, 11, 12, 13, 14)):
        return _ubx_product(Su, Fu.T, uv, 'ut')
    owner, generation = run['owner'], run['generation']
    P = int(Su.shape[1])
    keep = min(P, (((3 * P + 3) // 4) + 63) // 64 * 64)
    if keep == P:
        return _ubx_product(Su, Fu.T, uv, 'ut')
    cache = _O21_UPAIR_CACHE
    same = (cache.get('run') is run and cache.get('owner') is owner
            and cache.get('generation') == generation)
    if not same:
        cache.clear()
        if layer != 10:
            return _ubx_product(Su, Fu.T, uv, 'ut')
    mean = fnp.mean(Su, axis=0, dtype=fnp.float32)
    if not same:
        mean64 = mean.astype(fnp.float64)
        centered_energy = fnp.maximum(
            fnp.sum(Su * Su, axis=0, dtype=fnp.float64)
            - int(Su.shape[0]) * (mean64 * mean64), 0.0)
        factor_energy = fnp.sum(Fu * Fu, axis=0, dtype=fnp.float64)
        indices = fnp.sort(fnp.argsort(-(centered_energy * factor_energy), kind='stable')[:keep])
        # Fu is regenerated at the lazy join from the same unmodified owner.
        # Keep an owned selected factor for the complete owner lifetime.
        right = Fu[:, indices].T
        cache.update(run=run, owner=owner, generation=generation,
                     indices=indices, right=right, pairs=P, retained=keep)
        _O21_UPAIR_STATS['selected_generations'] += 1
    indices = cache['indices']
    left = Su[:, indices] - mean[indices]
    # A distinct leg token prevents collision with every original full-factor
    # cache entry. Within this token, owner+generation+fixedindices determine B.
    inner = _ubx_product(left, cache['right'], uv, 'o21_upair75_ut')
    full_mean = fnp.matmul(mean[None, :], Fu.T)
    inner = fnp.add(inner, full_mean, out=inner)
    _O21_UPAIR_STATS['calls'] += 1
    _O21_UPAIR_STATS['original_pairs'] += P
    _O21_UPAIR_STATS['retained_pairs'] += keep
    return inner
