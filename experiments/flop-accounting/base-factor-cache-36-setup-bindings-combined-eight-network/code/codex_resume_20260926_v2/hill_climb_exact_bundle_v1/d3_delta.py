

# Versioned overhead follow-up: prebind and fuse the nine existing depth-3
# star-product side keys through the unchanged setup binder and LVF kernels.
_PB_SINGLE = (
    (0, 3, 1024, 1024), (1, 3, 1024, 32), (2, 3, 1024, 32),
    (0, 3, 32, 1024), (1, 3, 1024, 1024), (2, 3, 32, 1024),
    (0, 3, 1024, 32), (1, 3, 32, 1024), (2, 3, 1024, 1024),
) + tuple(_PB_SINGLE)
