"""Target-free loader; existing rounded tape with certified actual final inputs."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def load(rank):
    tape = HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
    row = HERE/'rows_matrix_error_observer_v1'/f'row_{rank:04d}'
    assert json.loads((row/'result.json').read_text())['passed']
    data = {k: np.load(tape/f'{k}.npy', mmap_mode='r') for k in
            ('weights', 'pre_cov', 'sigma', 'x', 'alive')}
    with np.load(row/'final.npz') as f:
        data['x'] = data['x'].copy()
        data['sigma'] = data['sigma'].copy()
        data['x'][-1] = f['x']
        data['sigma'][-1] = f['sigma'].astype(np.float32)
        bank = f['prediction'][-1].copy()
    return data, bank
