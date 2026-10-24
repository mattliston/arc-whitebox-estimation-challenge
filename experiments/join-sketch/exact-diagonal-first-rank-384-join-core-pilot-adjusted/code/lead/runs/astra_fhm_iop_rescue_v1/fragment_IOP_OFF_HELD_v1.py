# HELD source-only fallback: disable IOP pruning; preserve exact full parent.
assert _ASTRA_IOP_ON is True
assert _ASTRA_DENSE_ON is True and _GTR_ENABLED is True and _ASTRA_FHM_ON is True
assert _ASTRA_IOP_CTX == [None] and not _ASTRA_IOP_CACHE
_ASTRA_IOP_ON = False
