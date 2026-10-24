"""Reuse incoming planes for the declared leading gate recurrence.

Callers pass the same FP32 g/d to this helper and carrier_fnp.gate. The
identity is exact in real arithmetic; native association error is separate.
"""


def post_planes(fnp, pre_planes, g, d):
    t21, q31, q22 = pre_planes
    g2 = g*g
    third = (g2[:, None]*g[None, :]*t21
             + (g*d)[:, None]*g[None, :]*q31
             + .5*g2[:, None]*d[None, :]*q22)
    fourth22 = g2[:, None]*g2[None, :]*q22
    return third, fourth22
