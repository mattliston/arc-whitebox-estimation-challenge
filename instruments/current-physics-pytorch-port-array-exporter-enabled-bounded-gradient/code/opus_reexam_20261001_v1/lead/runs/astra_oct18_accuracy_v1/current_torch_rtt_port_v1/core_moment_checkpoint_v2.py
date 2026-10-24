"""Pure bank moment checkpoints; explicit tensors only, no mutable card state."""
def install(cd, t):
    from torch.utils.checkpoint import checkpoint
    def diagonal_plain(pp, lam):
        p64 = pp.to(t.float64)
        return ((p64 * p64 * p64) * lam.to(t.float64)).sum(1)
    def columns_plain(pp, aa, beta):
        p64 = pp.to(t.float64)
        return (p64 * p64).T @ aa, p64.T @ beta
    def dispatch(fn, *args):
        if t.is_grad_enabled() and any(x.requires_grad for x in args):
            return checkpoint(fn, *args, use_reentrant=True, preserve_rng_state=False)
        return fn(*args)
    # Output coordinates are independent: diagonal reduces columns per row;
    # radial vectors reduce rows per column. Preserve each full reduction axis.
    def diagonal_chunks(pp, lam):
        return t.cat([dispatch(diagonal_plain, pp[i:i+64], lam)
                      for i in range(0, pp.shape[0], 64)], dim=0)
    def columns_chunks(pp, aa, beta):
        parts = [dispatch(columns_plain, pp[:, i:i+256], aa, beta)
                 for i in range(0, pp.shape[1], 256)]
        return t.cat([p[0] for p in parts]), t.cat([p[1] for p in parts])
    cd._o18_moment_diag = diagonal_chunks
    cd._o18_moment_columns = columns_chunks
    return diagonal_plain, columns_plain
