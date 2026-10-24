"""Read-only complete signed-pair cubic decomposition by birth cohort."""
import numpy as np

NAMES = ([f'mixed_birth_{l}' for l in range(15)] +
         [f'pure_birth_{l}' for l in range(15)] + ['coordinate_star', 'radial_trace'])


def split(pp, lam, counts):
    assert 2*sum(counts) == pp.shape[1] == len(lam)
    mixed, pure, offset = [], [], 0
    for count in counts:
        plus = pp[:, offset:offset+count].astype(np.float64)
        minus = pp[:, offset+count:offset+2*count].astype(np.float64)
        positive = lam[offset:offset+count].astype(np.float64)
        negative = lam[offset+count:offset+2*count].astype(np.float64)
        assert np.array_equal(positive, -negative)
        u, d = (plus+minus)/2, plus-minus
        mixed.append(3*np.sum(positive*u*u*d, axis=1))
        pure.append(.25*np.sum(positive*d*d*d, axis=1))
        offset += 2*count
    return np.stack(mixed+pure, axis=1)
