"""Export the fixed full90 first128 on-policy tensor correction onto the exact deployed 49bd/f938 parents.

Tensor-only: head, head_b, hidden_head and first_hidden_head rows 1..15 receive the sealed float32 addition
(identical arithmetic to fit_second_step_v1.deploy). Layer 0, every other tensor, every function and the
paid runtime tails stay byte-exact, so the runtime profile is the deployed #337506 profile.
usage: export_first128_onpolicy_tensors_v1.py <fit_receipts_dir> <out_dir>
"""
import ast, base64, hashlib, json, math, struct, subprocess, sys, zlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
NC = HERE.parent / 'runtime_first128_frozen_census_v1'
PARENTS = {'science': NC / 'candidate_first128_paid_nocensus_science_v1.py', 'release': NC / 'candidate_first128_paid_nocensus_release_v1.py'}
PINS = {'science': '49bd2ba31f0bce983f44cc001746e63ffaa84e530d3b31efa6424f69a0298c80', 'release': 'f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6'}
COLUMNS = (('head', 0, 64), ('head_b', 64, 65), ('hidden_head', 65, 193), ('first_hidden_head', 193, 321))
SHAPES = {'head': (16, 64), 'head_b': (16,), 'hidden_head': (16, 128), 'first_hidden_head': (16, 128)}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

def literal(text):
    tree = ast.parse(text)
    rows = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_GNN_LITERAL' for t in n.targets)]
    assert rows
    return ast.literal_eval(rows[-1]), tree

def unpack(item):
    shape, data = item
    raw = zlib.decompress(base64.b64decode(data)); assert len(raw) == 4 * math.prod(shape)
    values = np.frombuffer(raw, dtype='<f4').reshape(shape); assert np.isfinite(values).all()
    return raw, values

def functions(tree):
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

def main():
    receipts, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    fit = read(receipts / 'FITTED_v1.json'); seal = read(receipts / 'full90_COEFFICIENTS_v1.json')
    assert fit['successful'] and fit['passed'] and fit['status'] == 'FIVE_FOLD_FIRST128_ONPOLICY_PASS_NATIVE_REQUIRED'
    cv = fit['cross_validation']
    assert cv['mean_incremental_relative_pct'] <= -.15 and cv['incrementally_lower'] >= 54 and fit['negative_blocks'] >= 4
    assert all(f['physical']['passed'] for f in fit['families']) and fit['full90']['physical']['passed'] and fit['full90_fit_performed']
    assert fit['added_runtime_operations'] == 0 and fit['no_protected_truth_read'] and fit['all_targets_authentic_TRAIN90_only']
    assert fit['full90']['coefficient_seal_sha256'] == sha(receipts / 'full90_COEFFICIENTS_v1.json')
    coef = np.asarray(seal['coef_by_layer'], dtype='f8') * scale; assert scale > 0 and seal['family'] == 'full90' and len(seal['train']) == 90
    assert coef.shape == (15, 321) and np.isfinite(coef).all() and fit['final_coef_by_layer'] == seal['coef_by_layer']
    original = {}
    for role, path in PARENTS.items():
        assert sha(path) == PINS[role], role
        original[role] = path.read_text()
    parent = {role: literal(text)[0] for role, text in original.items()}; assert parent['science'] == parent['release']
    prior = parent['science']; updated = dict(prior); proofs = {}
    for key, lo, hi in COLUMNS:
        shape = tuple(prior[key][0]); assert shape == SHAPES[key], key
        raw, values = unpack(prior[key]); new = values.copy()
        new[1:] = (values[1:].astype('f8') + coef[:, lo:hi].reshape(values[1:].shape)).astype('f4')   # fitter deploy() arithmetic
        newraw = new.astype('<f4').tobytes(); assert len(newraw) == len(raw) and newraw[:4 * (hi - lo)] == raw[:4 * (hi - lo)]
        assert np.array_equal(new[0], values[0]) and np.isfinite(new).all()
        updated[key] = (prior[key][0], base64.b64encode(zlib.compress(newraw, 9)).decode())
        assert unpack(updated[key])[0] == newraw
        proofs[key] = dict(shape=list(shape), unchanged_layer0=True, original_sha256=hashlib.sha256(raw).hexdigest(), updated_sha256=hashlib.sha256(newraw).hexdigest(),
                           max_abs_change=float(np.abs(new.astype('f8') - values.astype('f8')).max()), float32_addition_to_existing_head=True)
    assert set(updated) == set(prior)
    for key in prior:
        if key not in SHAPES: assert updated[key] == prior[key]
    suffix = ('\n# Fixed one-step TRAIN90 first128 on-policy readout correction (strength scale %r); tensor-only,' % scale + ' exact 49bd/f938 physics and paid runtime tails.\n'
              'assert _GNN_PARAMS is None and not _E2_GNN_STATE\nassert not _FAST_RESCUE_FRAC\n_GNN_LITERAL = ' + repr(updated) + '\n_FAST_RESCUE_FRAC = {}\n')
    assert not out.exists(); out.mkdir(parents=True)
    files = {}
    for role, text in original.items():
        target = out / ('candidate_first128_onpolicy_second90_' + role + '_v1.py')
        target.write_text(text + suffix)
        written = target.read_text(); assert written.startswith(text)
        decoded, tree = literal(written); assert decoded == updated
        assert functions(tree) == functions(literal(text)[1])
        for key in prior:
            assert decoded[key][0] == prior[key][0]
            if key not in SHAPES: assert unpack(decoded[key])[0] == unpack(prior[key])[0]
        compile(written, str(target), 'exec')
        files[role] = dict(path=str(target), sha256=sha(target), bytes=target.stat().st_size, parent_sha256=PINS[role], entire_parent_prefix_exact=True, function_class_ast_exact=True)
    result = dict(utc=utc(), successful=True, strength_scale=scale, files=files, exporter_sha256=sha(__file__), fitted_sha256=sha(receipts / 'FITTED_v1.json'),
                  full90_seal_sha256=sha(receipts / 'full90_COEFFICIENTS_v1.json'), cross_validation=cv, negative_blocks=fit['negative_blocks'],
                  full90_physical=fit['full90']['physical'], tensor_proofs=proofs, all_other_tensors_and_shapes_exact=True, all_layer0_tensors_exact=True,
                  science_release_tensors_exact=True, added_runtime_operations=0, no_model_import_or_prediction=True, no_truth_arrays_read=True,
                  native_paired_mac32_vm32_vs_49bd_required=True, no_release_or_fire=True)
    (out / 'EXPORTED_v1.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(successful=True, files=files, cv=cv, negative_blocks=fit['negative_blocks'], max_abs_change={k: v['max_abs_change'] for k, v in proofs.items()})), flush=True)

if __name__ == '__main__': main()
