"""NumPy-free complete-source proof for six frozen on-policy diagnostic parents."""
import argparse, ast, base64, hashlib, importlib.util, json, subprocess, sys, types, zlib
from pathlib import Path

assert importlib.util.find_spec('numpy') is None
HERE = Path(__file__).resolve().parent
ROOT = Path('/home/arcworker/arc-workspace')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
PREREG = '0c94b0dc84bd17f0542df28e9ae56d7486c8944fe3cd08db74e6c91a76c24443'
MANIFEST = 'd6eaeb233f5f8109247d46cc68672848fb10477bd18a4076829fd267a4613136'

def definitions(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

def literal(source):
    return ast.literal_eval([n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == '_GNN_LITERAL' for t in n.targets)][-1])

def unpack(row):
    return zlib.decompress(base64.b64decode(row[1]))

def module(path, name):
    m = types.ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(path.read_text(), str(path), 'exec'), m.__dict__)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert sha(HERE/'PREREG_v1.json') == PREREG
    assert sha(HERE/'PARENT_SOURCES_v1.json') == MANIFEST
    pre = read(HERE/'PREREG_v1.json')
    manifest = read(HERE/'PARENT_SOURCES_v1.json')
    assert manifest['successful'] and manifest['full90_exact_b54_tensor_parity']
    assert manifest['prereg_sha256'] == PREREG
    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = object
    sys.modules['whestbench'] = stub
    full = manifest['families'][-1]
    assert full['key'] == 'full90'
    parents = {role: ROOT/full[role]['path'] for role in ('science', 'release')}
    parent_text = {role: path.read_text() for role, path in parents.items()}
    parent_lit = {role: literal(source) for role, source in parent_text.items()}
    parent_mod = {role: module(path, 'onpolicy_parent_'+role) for role, path in parents.items()}
    assert parent_lit['science'] == parent_lit['release']
    rows = []
    for j, family in enumerate(manifest['families']):
        key = family['key']
        held = pre['train'][j::5] if j < 5 else []
        train = [s for s in pre['train'] if s not in held]
        assert family['train'] == train and family['validation'] == held
        assert family['parent_fit_target_seed_ids'] == train
        assert family['parent_fit_excludes_validation'] and not set(train) & set(held)
        assert sha(ROOT/family['coef_path']) == family['coef_sha256']
        assert sha(ROOT/family['build_path']) == family['build_sha256']
        suffixes = []
        for role in ('science', 'release'):
            item = family[role]
            path = ROOT/item['path']
            assert sha(path) == item['sha256'] == item['effective_sha256']
            source = path.read_text()
            old = parent_text[role]
            assert source.startswith(old)
            suffixes.append(source[len(old):])
            assert definitions(source) == definitions(old)
            a, b = parent_lit[role], literal(source)
            assert set(a) == set(b)
            for name in a:
                assert a[name][0] == b[name][0]
                if name not in ('head', 'head_b', 'hidden_head'):
                    assert unpack(a[name]) == unpack(b[name])
            for name, count in [('head', 256), ('head_b', 4), ('hidden_head', 512)]:
                assert unpack(a[name])[:count] == unpack(b[name])[:count]
            if key == 'full90':
                assert source == old and b == a
            m = module(path, 'onpolicy_'+key+'_'+role)
            assert m._GNN_LITERAL == b
            assert m._GNN_PARAMS is None and not m._E2_GNN_STATE and m._FAST_RESCUE_FRAC == {}
            assert m._EDGE_PARAMETERS is None and m._MESSAGE_EDGE_PARAMETERS is None and not m._E2_EDGE_STATE
            assert m._O20_EDGE_HIDDEN == 8 and m._VMFR_CM == 8
            assert m._gn_message_edge_source_base is m._g5_message_edge_base
            assert not m._O20_FLATEXTRA_ON
            assert m._hzd_load is m._O20_FLATEXTRA_HZD_PARENT
            assert m._rs_load_b_owned_ql_v1 is m._O20_FLATEXTRA_OWNED_PARENT
            assert m.gnn_step.__globals__['_gop2_gnn_step'] is m._gop2_gnn_step
            assert m._gop2_gnn_step.__globals__['_hc90_hidden_head'] is m._hc90_hidden_head
            for name in ('_EDGE_LITERAL', '_MESSAGE_EDGE_LITERAL', '_O20_EDGE_HIDDEN', '_GN_EDGE',
                         '_O20_ECUT_RADICAL_A', '_O20_ECUT_RADICAL_B'):
                assert getattr(m, name) == getattr(parent_mod[role], name)
            marker = object()
            calls = []
            previous = m._gop2_gnn_step
            active = m._GOP2_ACTIVE[0]
            m._gop2_gnn_step = lambda *args: (calls.append(args) or marker)
            m._GOP2_ACTIVE[0] = True
            assert m.gnn_step(5, None, None, None, None, None) is marker and len(calls) == 1
            m._gop2_gnn_step = previous
            m._GOP2_ACTIVE[0] = active
            assert 'numpy' not in sys.modules
            rows.append(dict(family=key, role=role, source_sha256=sha(path), functions_exact=True,
                             only_existing_head_tensors_changed=True, cold_caches_empty=True,
                             live_bindings=True, excluded_validation=held))
        assert suffixes[0] == suffixes[1]
    result = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
                  successful=True, numpy_free=True, cases=rows, source_manifest_sha256=MANIFEST,
                  prereg_sha256=PREREG, script_sha256=sha(__file__), no_truth_arrays_read=True,
                  no_setup=True, no_prediction=True, no_capture_launched=True)
    with args.out.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(dict(successful=True, numpy_free=True, cases=len(rows))), flush=True)

if __name__ == '__main__':
    main()
