"""Independent dense-core and signed-cube identities for two nested moves."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20260919224)
f32 = lambda x: np.asarray(x, dtype=np.float32)
mm = lambda a, b: a @ b


def finite(a, name):
    assert a.dtype == np.float32 and np.isfinite(a).all(), name
    return a


env = dict(fnp=np, _shared_old384_state=lambda: {}, _shared_old384_push=None,
           _shared_old384_read=None, _shared_old384_mm=mm,
           _shared_old384_finite=finite, _mm=mm)
fragment = (HERE / 'nested224_fragment.py').read_text()
exec(compile(fragment, 'nested224_fragment.py', 'exec'), env)
env['_NESTED224_RANK'] = 7
r, n = 12, 16
F = f32(rng.normal(size=(r, 12)))
lam = f32([.4, .7, -.4, -.7, .3, .8, -.3, -.8, .5, .9, -.5, -.9])
state = dict(older_columns=0, older_joins=[], U2=None, F2=None, core2=None,
             F=F.copy(), core=F @ ((lam * lam)[:, None] * F.T),
             cohorts=[(0, 4), (4, 4), (8, 4)], columns=12)
rows = []
for layer in (8, 9):
    if layer == 9:
        O = f32(np.linalg.qr(rng.normal(size=(r, r)))[0] @ np.diag(np.linspace(.7, 1, r)))
        state['U2'] = O @ state['U2']
        state['F'] = O @ state['F']
        state['core'] = O @ state['core'] @ O.T
        extra = f32(rng.normal(size=(r, 4)))
        new_lam = f32([.6, 1.1, -.6, -1.1])
        state['F'] = np.concatenate([state['F'], extra], axis=1)
        state['core'] += extra @ ((new_lam * new_lam)[:, None] * extra.T)
        lam = np.concatenate([lam, new_lam])
        state['cohorts'].append((12, 4))
        state['columns'] = 16
    before = (state['F'].copy() if not state['older_columns'] else
              np.concatenate([state['U2'] @ state['F2'], state['F']], axis=1))
    env['_nested224_move'](state, layer, f32(rng.normal(size=(n, n))), lam)
    virtual = np.concatenate([state['U2'] @ state['F2'], state['F']], axis=1)
    cut = state['older_columns']
    expected = before.copy()
    # Independent expression for projection of each actual old column.
    expected[:, :cut] = state['U2'] @ state['U2'].T @ before[:, :cut]
    projection_rel = np.linalg.norm(virtual - expected) / np.linalg.norm(expected)
    core = virtual.astype(np.float64) @ ((lam.astype(np.float64)**2)[:, None] * virtual.astype(np.float64).T)
    core_rel = np.linalg.norm(state['core'] - core) / np.linalg.norm(core)
    Q = f32(rng.normal(size=(n, r)))
    state['pre_basis'] = Q
    pp = Q @ virtual
    live = np.arange(13)
    response = np.arange(9)
    got = env['_shared_old384_read'](pp[response], pp[live], lam, live, state)
    direct = (pp[response].astype(np.float64)**2 * lam.astype(np.float64)) @ pp[live].astype(np.float64).T
    plane_rel = np.linalg.norm(got - direct) / np.linalg.norm(direct)
    assert projection_rel < 2e-6 and core_rel < 2e-6 and plane_rel < 2e-6
    rows.append(dict(layer=layer, projection_relative_error=float(projection_rel),
                     full_core_relative_error=float(core_rel),
                     signed_cube_plane_relative_error=float(plane_rel)))
record = dict(passed=True, rows=rows,
              fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),
              reference='Explicit per-column orthogonal projection, weighted dense core, and signed-cube plane; no labels.')
with (HERE / 'nested224_certificate.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record), flush=True)
