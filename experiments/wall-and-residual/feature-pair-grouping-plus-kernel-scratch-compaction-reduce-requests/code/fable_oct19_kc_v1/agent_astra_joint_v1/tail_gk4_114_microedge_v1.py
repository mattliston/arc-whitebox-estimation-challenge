# Selected prespecified setting from the guarded four-network coadaptation screen.
# No fitting or use of the 25 judging ranks; full-stack validation is still required.
assert _GSK == 0.9 and _GK4 == 1.04
_GK4 = 1.14
# Constants depend on both coefficients but the inherited cache is keyed only by n.
# Import-time reset ensures setup constructs the selected coefficients.
_GQ33_CONST = {}
_GSK_CONST = {}
