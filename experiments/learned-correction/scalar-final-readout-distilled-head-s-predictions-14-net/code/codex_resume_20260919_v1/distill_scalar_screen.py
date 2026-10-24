"""Fixed teacher-only ridge: can a scalar final read retain the current head?

TRAIN100/150/200, CHECK35/36. The fit never reads truth; CHECK truth is used
only after its one payload is fixed. No change to the first fifteen layers.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / 'fable_wall_repair_v1/cand_c2_wallrepair_I95.py'
ROWS = P2 / 'fable_wall_repair_v1/rows_c2'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRAIN = (100, 150, 200)
CHECK = (35, 36)
MASKED = (13, 14, 15, 22, 23, 24, 25, 26, 27, 28)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hidden(x, state, w, p, masked):
    x = np.clip((x.astype(np.float32) - p['centers'][15]) / p['scales'][15], -8, 8)
    if masked:
        x[:, MASKED] = 0
    g = x[:, 5:6]
    src = state @ p['msg_w'].T + p['msg_b']
    cm = p['msg_w'].shape[0] // 3
    w2 = w * w
    w3 = w2 * w
    m1 = w.T @ src[:, :cm]
    m2 = w2.T @ src[:, cm:2*cm]
    m3 = w3.T @ src[:, 2*cm:]
    e1 = w2.T @ (src[:, :cm] ** 2)
    e2 = (w2*w2).T @ (src[:, cm:2*cm] ** 2)
    e3 = (w3*w3).T @ (src[:, 2*cm:] ** 2)
    msgs = np.concatenate([g*m1, g*m2, g*m3, np.log1p(np.maximum(e1, 0)),
                           np.log1p(np.maximum(e2, 0)), np.log1p(np.maximum(e3, 0))], axis=1)
    h = np.concatenate([x, np.clip(msgs, -8, 8)], axis=1)
    h = h @ p['enc0_w'].T + p['enc0_b']
    h = h / (1 + np.exp(-h))
    h = h @ p['enc2_w'].T + p['enc2_b']
    h = h / (1 + np.exp(-h))
    return h @ p['enc4_w'].T + p['enc4_b']


def main():
    for node in ast.parse(BASE.read_bytes()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == '_GNN_LITERAL':
            p = {k: np.asarray(v, dtype=np.float32) for k, v in ast.literal_eval(node.value).items()}
    manifest = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    row_data = {}
    for rank in TRAIN + CHECK:
        path = ROWS / f'rows_rank{rank:04d}.npz'
        weights = STORE / Path(manifest[rank]['path']).parent / 'weights.npy'
        with np.load(path) as rows:
            # Do not even decode TRAIN truth members.
            x = rows['x'][15]
            state = rows['state_in'][15]
            sigma = rows['sigma'][15].astype(np.float32)
            delta = rows['delta'][15].astype(np.float32)
            host = rows['host'][15]
        w = np.load(weights, mmap_mode='r')[15].astype(np.float32)
        full = hidden(x, state, w, p, False)
        cheap = hidden(x, state, w, p, True)
        gain = np.float32(1) + np.float32(.25) * np.tanh(p['gain'][15:16])
        scale = (sigma * p['rms'][15]) * gain
        replay = (sigma * p['rms'][15]) * (full @ p['head'][15] + p['head_b'][15]) * gain
        assert np.max(np.abs(replay-delta)) < 1e-9, (rank, np.max(np.abs(replay-delta)))
        row_data[rank] = dict(design=np.column_stack([cheap, np.ones(1024)]).astype(np.float64),
                             scale=scale.astype(np.float64), delta=delta.astype(np.float64), host=host,
                             source_sha256=sha(path), replay_max_abs=float(np.max(np.abs(replay-delta))))
    design = np.concatenate([row_data[r]['design'] for r in TRAIN])
    teacher = np.concatenate([row_data[r]['delta']/row_data[r]['scale'] for r in TRAIN])
    original = np.r_[p['head'][15], p['head_b'][15]].astype(np.float64)
    norms = np.maximum(np.sqrt(np.mean(design*design, axis=0)), 1e-12)
    normalized = design / norms
    residual = teacher - design @ original
    system = normalized.T @ normalized + len(design) * 1e-4 * np.eye(65)
    change = np.linalg.solve(system, normalized.T @ residual) / norms
    fitted = (original + change).astype(np.float32)
    out = HERE / 'scalar_distill_pilot_payload.npz'
    with out.open('xb') as stream:
        np.savez(stream, head=fitted[:64], head_b=fitted[64:], masked=np.asarray(MASKED))
    receipt = dict(source_sha256=sha(BASE), script_sha256=sha(__file__), train=list(TRAIN),
                   check=list(CHECK), fit_target='original current-head outputs only',
                   ridge=1e-4, payload_sha256=sha(out), rows={})
    for rank, row in row_data.items():
        pred_delta = (row['design'].astype(np.float32) @ fitted) * row['scale'].astype(np.float32)
        teacher_error = pred_delta.astype(np.float64) - row['delta']
        rec = dict(source_sha256=row['source_sha256'], replay_max_abs=row['replay_max_abs'],
                   teacher_error_mse=float(np.mean(teacher_error**2)),
                   teacher_energy=float(np.mean(row['delta']**2)))
        if rank in CHECK:
            with np.load(ROWS / f'rows_rank{rank:04d}.npz') as rows:
                truth = rows['truth'][15].astype(np.float64)
            old = np.maximum(row['host']+row['delta'], 0).astype(np.float32).astype(np.float64)
            proposed = np.maximum(row['host']+pred_delta.astype(np.float64), 0).astype(np.float32).astype(np.float64)
            rec.update(old_raw=float(np.mean((old-truth)**2)), proposed_raw=float(np.mean((proposed-truth)**2)))
            rec['raw_ratio'] = rec['proposed_raw']/rec['old_raw']
        receipt['rows'][str(rank)] = rec
    with (HERE / 'scalar_distill_pilot.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
