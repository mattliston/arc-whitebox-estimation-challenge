# E2 fuse edits: broadcast views ([:, None], .T) folded into einsum(optimize=False) products. numpy's einsum multiplies the operands
# of each output element left to right (tested bit-identical to the explicit broadcast products on both BLAS builds) and flopscope
# bills it exactly like the explicit multiplies; each saves the view requests.
LIVE = [
("""            K21 = cp_pre + star_pre + fnp.outer(vv if Bpre is Cpre else fnp.diag(Bpre), bpre) + Bpre * (2 * bpre)[:, None]
""", """            K21 = cp_pre + star_pre + fnp.outer(vv if Bpre is Cpre else fnp.diag(Bpre), bpre) + fnp.einsum('ij,i->ij', Bpre, 2 * bpre, optimize=False)
"""),
("""            _e2_gc = g[:, None]
            inherited = _e2_gc ** 2 * g * cp_pre
            bp = _e2_gc * Cpre
""", """            inherited = fnp.einsum('i,j,ij->ij', g ** 2, g, cp_pre, optimize=False)
            bp = fnp.einsum('i,ij->ij', g, Cpre, optimize=False)
"""),
("""            trace_post = fnp.outer(fnp.diag(Bpost), b) + Bpost * (2 * b)[:, None]
""", """            trace_post = fnp.outer(fnp.diag(Bpost), b) + fnp.einsum('ij,i->ij', Bpost, 2 * b, optimize=False)
"""),
("""            born = bp * bp * _e2_bb + (beta * births)[:, None] * bp.T
""", """            born = fnp.einsum('ij,ij,j->ij', bp, bp, _e2_bb, optimize=False) + fnp.einsum('i,ji->ij', beta * births, bp, optimize=False)
"""),
("""            _e2_ga = (g.astype(fnp.float32) * alive)[:, None]   # alive_i*(g_i*x) == (g_i*alive_i)*x exactly (alive is 0/1, g >= 0)
            p = _e2_ga * pp
""", """            _e2_ga = g.astype(fnp.float32) * alive   # alive_i*(g_i*x) == (g_i*alive_i)*x exactly (alive is 0/1, g >= 0)
            p = fnp.einsum('i,ij->ij', _e2_ga, pp, optimize=False)
"""),
]
OTHER = [
("""        ga = L['ga'] if L.get('ga_gate') is gate and L.get('ga_alive') is alive else (gate.astype(fnp.float32) * alive)[:, None]
        state['Q'] = ga * state['pre_basis']
""", """        ga = L['ga'] if L.get('ga_gate') is gate and L.get('ga_alive') is alive else gate.astype(fnp.float32) * alive
        state['Q'] = fnp.einsum('i,ij->ij', ga, state['pre_basis'], optimize=False)
"""),
("""    m2 = 2 * m
    m2c = m2[:, None]
    post21 = pk21 - m2c * cov
    post22 = pk22 - m2c * pk21.T - m2 * pk21 + fnp.outer(m2, m2) * cov - 2 * cov * cov
""", """    m2 = 2 * m
    post21 = pk21 - fnp.einsum('i,ij->ij', m2, cov, optimize=False)
    post22 = pk22 - fnp.einsum('i,ji->ij', m2, pk21, optimize=False) - m2 * pk21 + fnp.outer(m2, m2) * cov - 2 * cov * cov
"""),
("""    delta = ((2 * old * difference + difference * difference) * weight
             + quadratic_weight[:, None] * difference.T)
""", """    delta = ((2 * old * difference + difference * difference) * weight
             + fnp.einsum('i,ji->ij', quadratic_weight, difference, optimize=False))
"""),
("""    raw=(h@t['output'])*pf[:,None]*gf
""", """    raw=fnp.einsum('ij,i,j->ij',h@t['output'],pf,gf,optimize=False)
"""),
("""def gaussian_raw(mu, v):
    v = fnp.maximum(v, 1e-10)
    s = fnp.sqrt(v)
    a = mu / s
    pdf = fnp.exp(-a * a / 2) / fnp.sqrt(2 * 3.141592653589793)
    g = flops.stats.norm.cdf(a)
    m = s * pdf + mu * g
    r2 = (mu * mu + v) * g + mu * s * pdf
    r3 = mu * (mu * mu + 3 * v) * g + (mu * mu + 2 * v) * s * pdf
    r4 = (mu ** 4 + 6 * mu * mu * v + 3 * v * v) * g + mu * (mu * mu + 5 * v) * s * pdf
    return (s, a, pdf, g, m, r2, r3, r4)
""", """def gaussian_raw(mu, v):
    v = fnp.maximum(v, 1e-10)
    s = fnp.sqrt(v)
    a = mu / s
    pdf = fnp.exp(-a * a / 2) / 2.5066282746310002   # E2: = fnp.sqrt(2*pi) (correctly rounded), no request
    g = flops.stats.norm.cdf(a)
    m = s * pdf + mu * g
    mu2 = mu * mu   # E2: identical subexpressions computed once (same operations, same order)
    v3 = 3 * v
    r2 = (mu2 + v) * g + mu * s * pdf
    r3 = mu * (mu2 + v3) * g + (mu2 + 2 * v) * s * pdf
    r4 = (mu ** 4 + 6 * mu * mu * v + v3 * v) * g + mu * (mu2 + 5 * v) * s * pdf
    return (s, a, pdf, g, m, r2, r3, r4)
"""),
]
