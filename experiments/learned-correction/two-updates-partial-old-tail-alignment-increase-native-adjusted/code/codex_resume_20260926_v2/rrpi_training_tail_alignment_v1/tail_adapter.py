"""Target-free partial native BOTH-tail alignment; original join/card state retained.

The card's materialized full U/F supplies one block. Native lazy-coordinate
blocks and L10+ common256 reconstruction remain unmatched; this is not a
native-forward or gradient parity claim. Only the discrete row selector detaches.
"""
import inspect
import textwrap
from types import MethodType


def _tail_mask(t, layer, pp, U, F, lam, mu, vv, alive, old_columns):
    if not (7 <= layer <= 14 and old_columns):
        return pp, None
    assert U.shape[0] == pp.shape[0] and F.shape == (U.shape[1], old_columns)
    alpha = mu / vv.clamp_min(1e-10).sqrt()
    score = t.where((alpha < -2.5) & ~alive.bool(), alpha + 2.5, 2.5 - alpha)
    count = min(pp.shape[0], ((int((score.detach() >= 0).sum()) + 63) // 64) * 64)
    if count == pp.shape[0]:
        return pp, None
    order = t.argsort(-score.detach(), stable=True)
    tail = order[count:]
    keep = t.ones_like(alive, dtype=t.bool).index_fill(0, tail, False)
    pp = t.cat((t.where(keep[:, None], pp[:, :old_columns],
                       t.zeros_like(pp[:, :old_columns])), pp[:, old_columns:]), dim=1)
    u, factors = U.to(t.float64), F.to(t.float64)
    ut = u[tail]
    norm2 = (factors * factors).sum(0)
    rho = (u * u).sum() / (u.shape[0] * u.shape[1])
    repair_tail = (ut @ (factors @ (lam[:old_columns].to(t.float64) * norm2))) * (3 * rho)
    repair = t.zeros_like(mu).index_copy(0, tail, repair_tail)
    return pp, (tail, ut, factors, norm2, repair)


def _tail_radial(t, energy, column, aa, beta, tail_data):
    if tail_data is None:
        return energy, column
    tail, ut, factors, norm2, _ = tail_data
    variance = (ut * ut).sum(1) / ut.shape[1]
    de = norm2 * (aa[tail] * variance).sum()
    dc = (ut.T @ beta[tail]) @ factors
    n = factors.shape[1]
    return (t.cat((energy[:n] + de, energy[n:])),
            t.cat((column[:n] + dc, column[n:])))


def install(cd, torch):
    """Patch only this HookCard instance; no global hooks or persistent values."""
    original = cd.step.__func__
    source = textwrap.dedent(inspect.getsource(original))
    edits = (
        ("    omitted = alive * (1 - response_mask)\n",
         "    omitted = alive * (1 - response_mask)\n"
         "    pp, _tail_data = _tail_mask(t, layer, pp, U, F, lam, mu, vv, alive, old_columns)\n"),
        ("        cp = self.diagonal(cp, cpdiag)\n",
         "        if _tail_data is not None:\n"
         "            cpdiag = cpdiag + _tail_data[4]\n"
         "        cp = self.diagonal(cp, cpdiag)\n"),
        ("        column = p64.T @ beta\n",
         "        column = p64.T @ beta\n"
         "        energy, column = _tail_radial(t, energy, column, aa, beta, _tail_data)\n"),
    )
    for before, after in edits:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    namespace = dict(original.__globals__, _tail_mask=_tail_mask, _tail_radial=_tail_radial)
    exec(compile(source, __file__ + ':HookCard.step', 'exec'), namespace)
    cd.step = MethodType(namespace['step'], cd)
    return cd
