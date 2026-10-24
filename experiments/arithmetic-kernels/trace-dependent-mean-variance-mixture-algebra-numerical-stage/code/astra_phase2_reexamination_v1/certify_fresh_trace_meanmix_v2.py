"""Separate high-precision reference; v1 float raw-cumulant failure stays."""
from pathlib import Path
from decimal import Decimal, localcontext
from math import factorial
import numpy as np
import fresh_trace_meanmix_v1 as model

HERE = Path(__file__).resolve().parent


def mul(a, b):
    out = [Decimal(0)] * (len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def power(a, count):
    out = [Decimal(1)]
    for _ in range(count):
        out = mul(out, a)
    return out


def exact_scalar_raw(order, b, B, C, q):
    with localcontext() as ctx:
        ctx.prec = 80
        b, B, C, q = map(lambda z: Decimal(float(z)), (b, B, C, q))
        a = b/q
        V, D = B-b*b/q, C-B
        result = [Decimal(0)] * (order+1)
        for pairs in range(order//2+1):
            singles = order-2*pairs
            coefficient = Decimal(factorial(order)//(2**pairs*factorial(pairs)*factorial(singles)))
            polynomial = mul(power([-a, a], singles), power([D, V], pairs))
            for k, val in enumerate(polynomial):
                result[k] += coefficient*val
        moments = [Decimal(1)]
        for k in range(1, order+1):
            moments.append(moments[-1]*(1+(k-1)*q))
        return sum(x*y for x, y in zip(result, moments))


def high_precision_raw(indices, b, B, C, q):
    if len(b) != 1:
        return model.central_raw(indices, b, B, C, q)
    return exact_scalar_raw(len(indices), b[0], B[0, 0], C[0, 0], q)


def high_precision_directional(u, b, B, q):
    with localcontext() as ctx:
        ctx.prec = 80
        x, y, q = map(lambda z: Decimal(float(z)), (u@b, u@B@u, q))
        return {3: 3*x*y-x**3/q,
                4: 3*q*y*y+6*x*x*y-3*x**4/q,
                5: 30*q*x*y*y-6*x**5/q,
                6: 30*q*q*y**3+180*q*x*x*y*y-90*x**4*y}


source = (HERE/'certify_fresh_trace_meanmix_v1.py').read_text()
changes = (
    ("fresh_trace_meanmix_algebra_v1.json", "fresh_trace_meanmix_algebra_v2.json"),
    ("from fresh_trace_meanmix_v1 import central_raw, k3_tensor, k4_entry, planes, directional_cumulants",
     "from fresh_trace_meanmix_v1 import k3_tensor, k4_entry, planes\ncentral_raw = high_precision_raw\ndirectional_cumulants = high_precision_directional"),
    ("abs(actual[k]-expected[k])/(1e-13+abs(expected[k]))",
     "float(abs(actual[k]-expected[k])/(Decimal('1e-13')+abs(expected[k])))"),
    ("Exact finite Gamma-polynomial/Wick expectations, n4, twelve fixed directional checks per q.",
     "Exact finite Gamma-polynomial/Wick expectations, n4; separate 80-digit Decimal reference for twelve fixed directional checks per q. V1 float64 raw-cumulant cancellation failures remain archived.")
)
for before, after in changes:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
with localcontext() as context:
    context.prec = 80
    exec(compile(source, str(HERE/'certify_fresh_trace_meanmix_v1.py'), 'exec'),
         dict(__file__=str(HERE/'certify_fresh_trace_meanmix_v2.py'),
              high_precision_raw=high_precision_raw,
              high_precision_directional=high_precision_directional,
              Decimal=Decimal, __name__='__main__'))
