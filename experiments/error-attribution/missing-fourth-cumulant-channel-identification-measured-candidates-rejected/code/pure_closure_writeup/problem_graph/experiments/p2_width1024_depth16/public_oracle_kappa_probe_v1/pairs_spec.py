"""Fixed random unit-pair list shared by the pairs MC and the model tap.

2000 distinct unordered pairs (i < j) of units in [0, 1024), drawn with a fixed
seed so every consumer sees the same list.
"""

import numpy as np

PAIR_SEED = 20260825
N_PAIRS = 2000
WIDTH = 1024


def pair_list():
    rng = np.random.default_rng(PAIR_SEED)
    seen = set()
    pairs = []
    while len(pairs) < N_PAIRS:
        i, j = (int(v) for v in rng.integers(0, WIDTH, size=2))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((i, j))
    arr = np.array(pairs, dtype=np.int64)
    return arr[:, 0].copy(), arr[:, 1].copy()
