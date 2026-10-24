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
    cd._o18_moment_diag = lambda pp, lam: dispatch(diagonal_plain, pp, lam)
    cd._o18_moment_columns = lambda pp, aa, beta: dispatch(columns_plain, pp, aa, beta)
    return diagonal_plain, columns_plain
