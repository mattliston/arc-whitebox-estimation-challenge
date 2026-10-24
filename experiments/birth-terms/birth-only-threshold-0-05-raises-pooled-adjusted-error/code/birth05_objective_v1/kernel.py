

def _birth05_select(gate):
    return select(fnp.minimum(gate, 1 - gate), 0.05)
