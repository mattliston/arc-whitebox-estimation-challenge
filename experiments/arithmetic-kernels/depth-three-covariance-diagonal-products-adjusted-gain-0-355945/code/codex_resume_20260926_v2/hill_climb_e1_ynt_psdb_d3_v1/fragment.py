# Fixed D3 follow-on: five active P4 sides only; keep the D5 parent intact.
_PSD5_DEPTHS = {key: (3 if key in _PSD5_ACTIVE else 6) for key in _PSDB_KEYS}
_PSD5_REQUIRED = tuple((kind, 4 * 7**depth * (rows // (1 << depth)) * (cols // (1 << depth)))
                      for (kind, rows, cols), depth in _PSD5_DEPTHS.items())


def _psdb_group(A4, B4, out, offset, rows, cols):
    if ((offset, rows, cols) not in _PSD5_GROUPS
            or not all(_PSD5_DEPTHS.get(key) == 3 for key in
                       ((0, rows, 1024), (1, 1024, cols), (2, rows, cols)))):
        raise RuntimeError('unadmitted mixed-depth P4 symmetry group')
    return _PSD5_PARENT_GROUP(A4, B4, out, offset, rows, cols)
