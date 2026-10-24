# Runtime screen control: append the historical fourth-age coefficient .6.
_O20_ORIGPOLY_ADD_AGE4 = True
_O20_ORIGPOLY_PARENT_AGES = _np_ages
def _np_ages():
    ages = _O20_ORIGPOLY_PARENT_AGES()
    return ages + (0.6,) if _O20_ORIGPOLY_ADD_AGE4 else ages
