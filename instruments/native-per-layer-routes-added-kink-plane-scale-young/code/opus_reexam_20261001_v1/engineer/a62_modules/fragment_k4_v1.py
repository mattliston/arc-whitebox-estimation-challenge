# BEGIN OPUS-ENGINEER a62 K4 FAMILY (2026-10-12 00:1xZ; S50 arm fk4; trainer 10-11 23:5xZ): the torch card scales the radial k4 by band(l) +
# eps['k4'][l] at layers 1-14 BEFORE K4L's dk is added, and at layer 15 its 'k4_15' hook scales radial v^2 by (1 + eps['k4'][15]) with K4L's
# final term added unscaled. Natively, layers 1-14 are the literal _S19_K4BAND (one value per band 1-7 / 8-11 / 12-14, written by
# k4_from_pt_v1.py as the dials' band + the shift); layer 15 is this wrapper: the terminal scalar map's k4 = radial v^2 (1 + _K4_15) +
# _k4l_final(mu, v). _k4l_final is evaluated inside the terminal call's arguments, so its value arms the very next nonlinear_scalar call.
# _K4_15 = 0.0 = the parent bit for bit.
_K4_15 = 0.0
_K415_ARMED = [None]
_K415_PARENT_FINAL = _k4l_final
_K415_PARENT_SCALAR = nonlinear_scalar


def _k4l_final(mu, v):
    out = _K415_PARENT_FINAL(mu, v)
    _K415_ARMED[0] = out if _K4_15 else None
    return out


def nonlinear_scalar(mu, v, k3, k4):
    armed = _K415_ARMED[0]
    if armed is not None:
        _K415_ARMED[0] = None
        k4 = (k4 - armed) * (1.0 + _K4_15) + armed
    return _K415_PARENT_SCALAR(mu, v, k3, k4)
# END OPUS-ENGINEER a62 K4 FAMILY
