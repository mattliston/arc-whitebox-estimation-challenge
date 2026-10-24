"""One fixed, dimensionless block-normalized proposal direction; no fitting."""
import hashlib
import math

SEED = 's33_joint_direction_v1'
EPSILONS = (0.01, 0.005)
P_KEYS = ('head', 'head_b', 'msg_w', 'msg_b', 'enc0_w', 'enc0_b',
          'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')
DIAL_KEYS = ('X1', 'X0', 'X2', 'bscale', 's_carry', 's_new', 'c0', 'c1',
             'k4a', 'k4b', 'k4c')

def signs(np, name, size):
    return np.asarray([1.0 if hashlib.sha256(
        (SEED + ':' + name + ':' + str(i)).encode()).digest()[0] & 1 else -1.0
        for i in range(size)], dtype=np.float64)

def make_directions(np, arrays, dials):
    active = {'p.' + k for k in P_KEYS} | {'a', 'c'} | {
        k for k in arrays if k.startswith('e.')}
    assert len(active) == 23 and active <= set(arrays)
    assert set(DIAL_KEYS) <= set(dials)
    denom = math.sqrt(len(active) + len(DIAL_KEYS))
    directions, records = {}, {}
    for name, original in arrays.items():
        v = np.asarray(original, dtype=np.float64)
        direction = np.zeros_like(v)
        if name in active:
            direction = signs(np, name, v.size).reshape(v.shape)
            if name == 'p.head': direction[[0, 15], :] = 0
            if name == 'p.head_b': direction[[0, 15]] = 0
            if name == 'c': direction[:, [1, 3]] = 0
            scale = max(float(np.sqrt(np.mean(v * v))), 1e-3)
            direction *= scale / (denom * float(np.sqrt(np.mean(direction * direction))))
        directions[name] = direction
        records[name] = {'shape': list(v.shape), 'active': name in active,
                         'direction_rms': float(np.sqrt(np.mean(direction * direction))),
                         'sha256_f64': hashlib.sha256(direction.tobytes()).hexdigest()}
    dd = {}
    for name, value in dials.items():
        dd[name] = (float(signs(np, 'dial.' + name, 1)[0]) * max(abs(float(value)), 1e-3)
                    / denom) if name in DIAL_KEYS else 0.0
    return directions, dd, {'seed': SEED, 'block_count': 34, 'arrays': records, 'dials': dd,
                           'no_terminal_head_direction': True,
                           'fixed_parameter_groups': ['centers', 'scales', 'rms', 'gain', 'first']}

def perturb(np, arrays, dials, directions, dial_directions, epsilon):
    return ({k: (np.asarray(v, dtype=np.float64) + epsilon * directions[k]).astype(np.float32)
             for k, v in arrays.items()},
            {k: float(v) + epsilon * dial_directions[k] for k, v in dials.items()})
