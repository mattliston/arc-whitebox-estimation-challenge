

# BEGIN SCOUT-NATIVE F32 ZERO FILLS v1 (2026-10-12; byte-identical): the strip zero fills of the tree kernel (_hz_leaf's fill slices and
# _hzd_load's two zero strips) passed a Python 0.0, which flopscope bills at the float64 rate (2 flops an element, measured) although
# the targets are float32; a float32 0-d zero bills 1 (measured). The same zeros land in the same places. About -0.034% of bill
# (census of the S50 twin: _hz_leaf:10212 0.0637%, _hzd_load:13112/13114 0.0045%, all at the float64 rate).
_F32Z = [None]


def _f32z():
    z = _F32Z[0]
    if z is None:
        z = _F32Z[0] = fnp.zeros((), dtype=fnp.float32)
    return z


def _hz_leaf(v):
    h = _BH_HOLD.append
    for x, y, o in v['mats']:
        h(fnp.matmul(x, y, out=o))
    z = _f32z()
    for f in v['fills']:
        h(fnp.copyto(f, z))


def _hzd_load(quad, x, L, rm, cm, hr, hc):
    # = the parent's _hzd_load with float32 zero fills
    main, rs, cs, cr, rz, cz = _hzd_parts(quad, L, rm, cm, hr, hc)
    h = _BH_HOLD.append
    u, s = 1 << L, 1 << (L - 1)
    R, C = u * rm, u * cm
    xm = x[:R, :C]
    vm = fnp.reshape(xm, (2,) * L + (rm,) + (2,) * L + (cm,))
    h((xm, vm)); h(fnp.copyto(main, vm))
    if hr:
        xr = x[R:, :C]
        vr = fnp.reshape(xr, (2,) * (L - 1) + (2,) * L + (cm,))
        h((xr, vr)); h(fnp.copyto(rs, vr))
    if hc:
        xc = x[:R, C:]
        vc = fnp.reshape(xc, (2,) * L + (rm,) + (2,) * (L - 1))
        h((xc, vc)); h(fnp.copyto(cs, vc))
    if hr and hc:
        xx = x[R:, C:]
        vx = fnp.reshape(xx, (2,) * (L - 1) + (2,) * (L - 1))
        h((xx, vx)); h(fnp.copyto(cr, vx))
    z = _f32z()
    if hr:
        h(fnp.copyto(rz, z))
    if hc:
        h(fnp.copyto(cz, z))
# END SCOUT-NATIVE F32 ZERO FILLS v1
