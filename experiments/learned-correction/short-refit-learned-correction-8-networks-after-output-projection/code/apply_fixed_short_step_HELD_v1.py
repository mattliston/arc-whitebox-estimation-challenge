"""One predetermined RAD-style half trust step, never a sweep or line search."""
import argparse, ast, base64, hashlib, json, math, sys, zlib
from pathlib import Path
D = Path(__file__).resolve().parent; O = D.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, v):
    with Path(p).open('x') as f: json.dump(v, f, indent=2); f.write('\n')
def main():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True); p.add_argument('--retirement', type=Path, required=True)
    p.add_argument('--retirement-sha256', required=True); a = p.parse_args()
    assert sha(a.plan) == a.plan_sha256 and sha(a.retirement) == a.retirement_sha256
    c, accepted = json.loads(a.plan.read_text()), json.loads(a.retirement.read_text())
    assert c['phase'] == 'train' and accepted['approved'] is True and accepted['reviewed_by'] == 'root'
    assert accepted['plan_sha256'] == a.plan_sha256 and accepted['all_owned_processes_absent'] is True
    for path, h in c['pins'].items(): assert sha(path) == h, path
    rows, custody = [], []
    for shard in (0, 1):
        run = D/(c['run_prefix']+'_s'+str(shard)+'_v1')
        ready = run/'PHASE_COMPLETE_HELD_v1.json'; ready_hash = sha(ready)
        assert accepted['phase_complete_pins'][str(ready)] == ready_hash
        phase = json.loads(ready.read_text()); assert phase['plan_sha256'] == a.plan_sha256
        for e in phase['evidence']:
            case = e['case']; assert case in c['fixed_calls'] and case['shard'] == shard
            assert sha(e['result']) == e['result_sha256'] and sha(e['claim']) == e['claim_sha256']
            r = torch.load(e['result'], map_location='cpu', weights_only=False)
            assert r['rank'] == case['seed'] and r['mode'] == 'grad' and r['scale'] == 1e8
            assert r['params'] == c['params']['base'] and r['threads'] == 1 and r['card'] == 'M'
            assert math.isfinite(r['mse']) and r['mse'] > 0
            rows.append(r); custody.append(e)
    assert sorted(r['rank'] for r in rows) == list(range(7100268, 7100276)) and len(rows) == 8
    rows.sort(key=lambda r: r['rank'])
    saved = torch.load(c['params']['base'], map_location='cpu', weights_only=True)
    arrays = {k: v.clone() for k, v in saved['arrays'].items()}
    dials = dict(saved['dials']); pl = saved['pl']
    mse = sum(r['mse'] for r in rows)/8 * 1e8
    directions, report, first = {}, {}, 0.0
    for key in ('p.head', 'p.head_b'):
        G = torch.stack([r['grads'][key].double()*(1e8/r['scale']) for r in rows])
        assert bool(torch.isfinite(G).all()) and tuple(G.shape[1:]) == tuple(arrays[key].shape)
        m, se = G.mean(0), G.std(0)/math.sqrt(8)
        t = torch.where(se > 0, m/se.clamp_min(1e-300), torch.zeros_like(m))
        keep = (t.abs() > 2.5).double(); rms = float(arrays[key].double().square().mean().sqrt())
        directions[key] = -torch.sign(m)*keep*rms
        amount = float((m.abs()*(1-1/(t*t).clamp_min(1e-12))*keep).sum()*rms)
        first += amount; report[key] = dict(moving=int(keep.sum()), total=m.numel(), rms=rms, corrected_first_order=amount)
    head_size = 0.5*min(0.005*mse/max(first, 1e-300), 0.002)
    for key, direction in directions.items(): arrays[key] = (arrays[key].double()+head_size*direction).float()
    for key in ('X1', 'bscale', 'c0'):
        slopes = [float(r['dial_grads'][key].double())*(1e8/r['scale']) for r in rows]
        assert all(math.isfinite(x) for x in slopes)
        slope = sum(slopes)/8
        shift = -float(np.sign(slope))*0.5*min(0.005*mse/max(abs(slope), 1e-300), 0.02)
        dials[key] += shift; report[key] = dict(slope=slope, shift=shift)
    assert set(k for k in arrays if not torch.equal(arrays[k], saved['arrays'][k])) <= {'p.head', 'p.head_b'}
    assert set(k for k in dials if dials[k] != saved['dials'][k]) <= {'X1', 'bscale', 'c0'}
    point = D/'point_v1'; point.mkdir(exist_ok=False)
    with (point/'params.pt').open('xb') as f: torch.save(dict(arrays=arrays, dials=dials, pl=pl), f)
    (point/'hu.json').write_bytes(Path(c['hu_path']).read_bytes())
    # Exact four-expression graft onto the current native candidate. All other bytes are preserved.
    parent = Path(c['native_parent']); raw = parent.read_bytes(); text = raw.decode(); nodes = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nodes.setdefault(node.targets[0].id, []).append(node)
    literal = ast.literal_eval(nodes['_GNN_LITERAL'][0].value)
    for key in ('head', 'head_b'):
        value = arrays['p.'+key].double().numpy()
        scale = 1+np.asarray(pl['ds'], dtype=np.float64)
        value = np.asarray(value*(scale[:, None] if key == 'head' else scale), dtype=np.float32)
        literal[key] = (tuple(value.shape), base64.b64encode(zlib.compress(value.tobytes())).decode())
    for key in ('X1', 'bscale', 'c0'): assert len(set(pl[key])) == 1
    old_c0_c1 = ast.literal_eval(nodes['_K31R_TABLE'][-1].value.value)
    replacements = [
        ('_GNN_LITERAL', nodes['_GNN_LITERAL'][0].value, literal),
        ('_BD_J1_PHYS', nodes['_BD_J1_PHYS'][-1].value, dials['X1']+pl['X1'][0]),
        ('_S19_BSCALE', nodes['_S19_BSCALE'][-1].value, dials['bscale']+pl['bscale'][0]),
        ('_K31R_TABLE', nodes['_K31R_TABLE'][-1].value.value, (dials['c0']+pl['c0'][0], old_c0_c1[1]))]
    starts = [0]
    for line in raw.splitlines(keepends=True): starts.append(starts[-1]+len(line))
    edits = sorted((starts[n.lineno-1]+n.col_offset, starts[n.end_lineno-1]+n.end_col_offset, name, repr(value).encode()) for name, n, value in replacements)
    pieces, undo, cursor = [], [], 0
    for lo, hi, name, value in edits:
        pieces.extend((raw[cursor:lo], value)); undo.extend((raw[cursor:lo], raw[lo:hi])); cursor = hi
    result = b''.join(pieces+[raw[cursor:]])
    assert b''.join(undo+[raw[cursor:]]) == raw
    compile(result, str(point/'candidate.py'), 'exec')
    (point/'candidate.py').write_bytes(result)
    write(point/'FIXED_STEP_v1.json', dict(training_plan_sha256=a.plan_sha256, root_retirement_sha256=a.retirement_sha256, training_seeds=[r['rank'] for r in rows], training_mse=mse/1e8, head_component_size=head_size, head_corrected_first_order=first, components=report, fixed_half_damping=0.5, train_selection=False, check_selection=False, parameter_pins={str(point/f): sha(point/f) for f in ('params.pt', 'hu.json', 'candidate.py')}, source_changed_expressions=[v[0] for v in replacements], native_parent_sha256=sha(parent), native_other_bytes_identical=True, source_reuses_current_operations=True, actual_native_bill_required=True, custody=custody))
    print(json.dumps(dict(point=str(point), candidate_sha256=sha(point/'candidate.py'), step_sha256=sha(point/'FIXED_STEP_v1.json'))))
if __name__ == '__main__': main()
