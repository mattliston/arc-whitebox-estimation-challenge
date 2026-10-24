# BEGIN OPUS-ENGINEER EDGE TANH STAND-IN (a17 "ets", 2026-10-02; an approximation of one learned activation to 2e-4, not bit-exact).
# The learned covariance edge (layers 8-14) applies tanh to an (n, n, 16) float32 tensor: 16 flops per element, 0.715% of the bill.
# Stand-in on honestly priced pointwise ops: tanh(x) = s / sqrt(1 + s^2) with s = sinh(x) ~ x (1 + c1 u + c2 u^2), u = min(x^2, 81)
# (the clamp keeps every intermediate finite for |x| up to 1e15; beyond u = 81 the quotient is 1 to 1e-6). Largest absolute error
# against tanh 2.0e-4 (a17_tanh/probe_tanh.py). Written so that one scratch array suffices and no constant multiply is paid:
# 1 + c1 u + c2 u^2 = c2 ((u + A)^2 + B), and with S = x ((u + A)^2 + B):  tanh(x) ~ S / sqrt(S^2 + 1 / c2^2).
# Ten in-place passes, 10 flops per element instead of 16.
_ETS_C1 = 0.165
_ETS_C2 = 0.0098
_ETS_A = _ETS_C1 / (2 * _ETS_C2)
_ETS_B = 1 / _ETS_C2 - _ETS_A ** 2
_ETS_K = 1 / _ETS_C2 ** 2
_ETS_SCRATCH = {}


def _ets_tanh(h, hold):
    """h <- the stand-in for tanh(h), in place; returns the alias of h. hold: list that keeps the out= aliases to the caller's exit."""
    shape = tuple(int(x) for x in h.shape)
    w = _ETS_SCRATCH.get(shape)
    if w is None:
        w = _ETS_SCRATCH[shape] = fnp.empty(shape, dtype=fnp.float32)
    hold.append(fnp.multiply(h, h, out=w))
    hold.append(fnp.minimum(w, 81.0, out=w))
    hold.append(fnp.add(w, _ETS_A, out=w))
    hold.append(fnp.multiply(w, w, out=w))
    hold.append(fnp.add(w, _ETS_B, out=w))
    hold.append(fnp.multiply(h, w, out=h))
    hold.append(fnp.multiply(h, h, out=w))
    hold.append(fnp.add(w, _ETS_K, out=w))
    hold.append(fnp.sqrt(w, out=w))
    return fnp.divide(h, w, out=h)
# END OPUS-ENGINEER EDGE TANH STAND-IN
