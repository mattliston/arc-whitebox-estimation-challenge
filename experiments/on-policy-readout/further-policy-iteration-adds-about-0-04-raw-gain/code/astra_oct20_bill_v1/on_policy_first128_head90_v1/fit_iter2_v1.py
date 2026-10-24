"""Lean iteration-2 fit: 705-column on-policy readout refit on captures of the FIRED stack's own trajectory (TRAIN90, no folds).
Reuses the frozen loader (with the extended delta so 'old' matches the saved delta) and the gate fitter's design/solver.
usage: ARC_RIDGE=0.1 ARC_DAMP=0.5 fit_iter2_v1.py <capture_run_dir> <teacher_science.py> <out_dir>
"""
import glob, gzip, hashlib, importlib.util, inspect, json, os, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fs', HERE / 'fit_second_step_v1.py'); fs = importlib.util.module_from_spec(spec); spec.loader.exec_module(fs)
spec2 = importlib.util.spec_from_file_location('fg', HERE / 'fit_extended_gate_v1.py'); fg = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fg)
cf, f, N, W0, W1, NEW = fs.cf, fs.f, fs.N, fs.WIDTH, fg.W1, fg.NEW
cf.RIDGE = float(os.environ.get('ARC_RIDGE', '0.1')); cf.DAMPING = float(os.environ.get('ARC_DAMP', '0.5'))
sha, read, utc, write = fs.sha, fs.read, fs.utc, fs.write

def delta32_ext(P, layer, state, hidden, first, sigma, gate=None):
    ss, hh, ff, sig = [np.asarray(v, dtype='f4') for v in (state, hidden, first, sigma)]
    inner = ((ss @ P['head'][layer] + hh @ P['hidden_head'][layer]) + ff @ P['first_hidden_head'][layer])
    inner = (inner + (hh * hh) @ P['sq_hidden_head'][layer]) + (ff * ff) @ P['sq_first_head'][layer]
    if gate is None: raise ValueError('gate required')
    inner = inner + (ff * np.asarray(gate, dtype='f4')[:, None]) @ P['gate_first_head'][layer]
    inner = inner + P['head_b'][layer]
    gain = np.float32(1) + np.float32(.25) * np.tanh(P['gain'][layer:layer + 1])
    result = ((sig * P['rms'][layer]) * inner) * gain
    assert result.dtype == np.dtype('f4') and np.isfinite(result).all()
    return result
fs.delta32_ext = delta32_ext
_src = inspect.getsource(fs.data_rows)
assert _src.count('old = delta32(P,layer,state,hidden,first,sigma)') == 1 and _src.count('physical=physical,') == 1
_src = _src.replace('old = delta32(P,layer,state,hidden,first,sigma)', 'old = delta32_ext(P,layer,state,hidden,first,sigma,gate)').replace('physical=physical,', 'physical=physical,gate=gate,').replace('def data_rows(', 'def data_rows_iter2(')
_src = _src.replace("assert role in ('TRAIN','HELD')", "assert role in ('TRAIN','HELD')").replace("if role == 'HELD':", "if False:")
exec(_src, fs.__dict__); data_rows = fs.data_rows_iter2

def deploy_iter2(P, coef):
    assert coef.shape == (15, W1)
    out = fs.deploy(P, coef[:, :W0])
    for key, lo in ((NEW[0], W0), (NEW[1], W0 + 128), (NEW[2], W0 + 256)):
        t = P[key].copy(); t[1:] = (P[key][1:].astype('f8') + coef[:, lo:lo + 128]).astype('f4'); out[key] = t
    return out

def evaluate(rows, P, Q, seeds):
    result = []; numerator = np.zeros(15); denominator = np.zeros(15); previous = None; original = loss = 0.0
    for r in rows:
        seed, layer = r['seed'], r['layer']; l = layer - 1
        if seed != previous:
            assert layer == 1; previous, original, loss = seed, 0.0, 0.0
        new = delta32_ext(Q, layer, r['state'], r['hidden'], r['first'], r['sigma'], r['gate'])
        change = (new.astype('f8') - r['old'].astype('f8')) * r['mask']; correction = change / r['physical']
        original += float(r['yA'] @ r['yB']); loss += float((r['yA'] - correction) @ (r['yB'] - correction))
        numerator[l] += float(change @ change); masked = r['saved'] * r['mask']; denominator[l] += float(masked @ masked)
        if layer == 15:
            result.append(dict(seed=seed, incremental_pct=100 * (loss / original - 1)))
    assert [r['seed'] for r in result] == list(seeds) and np.all(denominator > 0)
    ratios = np.sqrt(numerator / denominator); overall = float(np.sqrt(numerator.sum() / denominator.sum()))
    return result, dict(correction_to_parent_head_rms=overall, per_layer_ratios=ratios.tolist(), max_layer_ratio=float(ratios.max()), passed=bool(overall <= .25 and ratios.max() <= 1))

def main():
    run, teacher, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    tsha = sha(teacher); lit, _ = f.literals_and_functions(teacher.read_text()); P = f.decode(lit['_GNN_LITERAL'])
    assert all(k in P for k in NEW) and P['first_hidden_head'].shape == (16, 128)
    design = read(fs.D / 'DESIGN_LOCAL_v1.json'); plan = read(fs.D / 'PLAN_CAPTURE_v1.json')
    files, preds = {}, {}
    for part in sorted(run.glob('part*/')):
        meta = part / 'gnn_full_state_v1' / 'METADATA_v1.json'
        if not meta.exists(): continue
        assert read(meta)['source_sha256'] == tsha
        for p in sorted((part / 'gnn_full_state_v1').glob('rank*_scored_L*.json.gz')):
            name = p.name; seed = int(name[4:11]); layer = int(name.split('_L')[1][:2]); files[(seed, layer)] = dict(path=str(p), sha256=sha(p))
        for p in sorted(part.glob('diagnostic_pred_*.bin')):
            seed = int(p.stem.split('_')[-1]); preds[seed] = dict(path=str(p))
    seeds = [s for s in design['train'] if s in preds and all((s, l) in files for l in range(1, 16))]
    assert len(seeds) >= 60, len(seeds); print(json.dumps(dict(utc=utc(), captured_seeds=len(seeds))), flush=True)
    info = dict(family=dict(key='iter2', train=seeds, validation=[], science=dict(sha256=tsha)), files=files, predictions=preds, P=P)
    G, b = fg.moments(data_rows(seeds, info, design, plan, 'TRAIN'), seeds)
    coef = fg.coefficients(G, b, len(seeds), P); Q = deploy_iter2(P, coef); del G, b
    rows, trust = evaluate(fg.extend(data_rows(seeds, info, design, plan, 'TRAIN')), P, Q, seeds)
    out.mkdir(exist_ok=False)
    seal = out / 'full90_COEFFICIENTS_v1.json'
    write(seal, dict(utc=utc(), family='iter2', train=seeds, coef_by_layer=coef.tolist(), physical=trust, width=W1, teacher_sha256=tsha, ridge=cf.RIDGE, damping=cf.DAMPING))
    summary = dict(n=len(rows), mean_incremental_relative_pct=float(np.mean([r['incremental_pct'] for r in rows])), incrementally_lower=int(sum(r['incremental_pct'] < 0 for r in rows)))
    write(out / 'FITTED_v1.json', dict(utc=utc(), successful=True, status='ITER2_TRAIN_ONLY_NATIVE_REQUIRED', width=W1, teacher_sha256=tsha, training=summary, physical=trust, seal_sha256=sha(seal), no_cv=True, ridge=cf.RIDGE, damping=cf.DAMPING))
    print(json.dumps(dict(utc=utc(), training=summary, physical=dict(rms=trust['correction_to_parent_head_rms'], max_layer=trust['max_layer_ratio'], passed=trust['passed']))), flush=True)

if __name__ == '__main__': main()
