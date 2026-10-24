"""Small exact block-ridge model for a network-level final correction."""
from pathlib import Path
import hashlib
import json
import numpy as np

D = Path(__file__).resolve().parent
H = D.parent
P2 = H.parent
G = H / 'native_final_graph_head_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def arrays(path, **data):
    with path.open('xb') as stream:
        np.savez(stream, **data)


def bind():
    cfg = json.loads((D/'protocol.json').read_text())
    seal = json.loads((D/'source_seal.json').read_text())
    for name, value in seal['files'].items():
        assert sha(D/name) == value, name
    assert sha(G/'capture_complete.json') == cfg['capture_sha256']
    capture = json.loads((G/'capture_complete.json').read_text())
    assert capture['all_groups_retired'] and capture['swap_growth_bytes'] == 0
    return cfg, {row['rank']: row for row in capture['rows']}


def block_system(z, b, y):
    """z[n,p], b[n,m,k], y[n,m]; order (feature, response mode)."""
    n, m, k = b.shape
    assert z.shape[0] == n and y.shape == (n, m)
    gram = np.einsum('nma,nmb->nab', b, b, optimize=True) / m
    cross = np.einsum('nma,nm->na', b, y, optimize=True) / m
    system = np.einsum('np,nq,nab->paqb', z, z, gram, optimize=True)
    system = system.reshape(z.shape[1]*k, z.shape[1]*k) / n
    rhs = np.einsum('np,na->pa', z, cross, optimize=True).reshape(-1) / n
    penalty = np.ones(z.shape[1]*k)
    penalty[:k] = 0
    system[np.diag_indices_from(system)] += penalty
    coeff = np.linalg.solve(system, rhs).reshape(z.shape[1], k)
    residual = np.linalg.norm(system @ coeff.ravel()-rhs) / max(np.linalg.norm(rhs), 1e-30)
    assert residual < 1e-10
    return coeff, float(residual)


def response(control, sigma):
    alpha, gate, pdf = control[:, :, 3], control[:, :, 5], control[:, :, 6]
    phi = np.stack((gate, alpha*gate, pdf, alpha*pdf, (alpha*alpha-1)*pdf), axis=2)
    return sigma[:, :, None]*.001*phi


def load(ranks, records, cfg, labels=False, allowed_labels=()):
    if labels:
        assert list(ranks) == list(allowed_labels)
    xs, ss, bs, ts, provenance = [], [], [], [], []
    for rank in ranks:
        assert rank in cfg['ranks']
        assert rank not in cfg['excluded_check32'] and rank not in cfg['excluded_valid40']
        record = records[rank]
        packet = G/record['path']
        assert sha(packet) == record['sha256']
        with np.load(packet) as saved:
            xs.append(saved['control'].astype(np.float64))
            ss.append(saved['sigma'].astype(np.float64))
            bs.append(saved['prediction'][15].astype(np.float64))
        if labels:
            path = P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
            with np.load(path) as saved:
                ts.append(saved['mc_post'][15].astype(np.float64))
            provenance.append(dict(rank=rank, sha256=sha(path)))
    x, sigma, base = np.stack(xs), np.stack(ss), np.stack(bs)
    assert x.shape == (len(ranks),1024,94) and sigma.shape == base.shape == (len(ranks),1024)
    assert all(np.isfinite(a).all() for a in (x,sigma,base)) and (sigma > 0).all()
    return x, sigma, base, np.stack(ts) if labels else None, provenance
