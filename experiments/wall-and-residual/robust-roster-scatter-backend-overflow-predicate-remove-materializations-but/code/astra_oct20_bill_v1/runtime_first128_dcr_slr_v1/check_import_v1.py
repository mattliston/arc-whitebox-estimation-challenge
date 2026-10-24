"""NumPy-free complete import proof for the first128 paid runtime plus robust DCR source, no native."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
assert importlib.util.find_spec('numpy') is None
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((HERE / 'BUILD_v1.json').read_text())
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub


def local(path):
    path = Path(path)
    if path.exists():
        return path
    prefix = '/Users/matt/arc-white-box-estimation/'
    assert str(path).startswith(prefix)
    return Path('/home/arcworker/arc-workspace') / str(path).removeprefix(prefix)


def module(path, name):
    result = types.ModuleType(name); result.__file__ = str(path); sys.modules[name] = result
    exec(compile(path.read_text(), str(path), 'exec'), result.__dict__)
    return result


parents = {}; cases = []
for key, roles in build['outputs'].items():
    for role, rec in roles.items():
        path, parent = local(rec['path']), local(rec['parent_path'])
        assert sha(path) == rec['sha256'] and sha(parent) == rec['parent_sha256']
        parent_key = (rec['parent_sha256'], role)
        if parent_key not in parents:
            parents[parent_key] = module(parent, 'smoff_parent_' + key + '_' + role)
        old = parents[parent_key]
        assert sha(old.__file__) == rec['parent_sha256']
        current = module(path, 'smoff_' + key + '_' + role)
        assert path.read_bytes().startswith(parent.read_bytes())
        assert ast.unparse(ast.parse(path.read_text()).body[-1]) == '_FAST_RESCUE_FRAC = {}'
        assert current._FAST_RESCUE_FRAC == {} and current._O20_SMDIRECT_ON is False
        assert old._O20_SMDIRECT_ON is False
        assert current._O20_NPX_COPY_EXACT_ON is False and old._O20_NPX_COPY_EXACT_ON is False
        assert current._dcr_right.__code__.co_code == old._dcr_right.__code__.co_code
        assert current._dcr_right.__code__.co_consts == old._dcr_right.__code__.co_consts
        assert current._dcr_right.__code__.co_names == old._dcr_right.__code__.co_names
        assert current._O21_SLR_CHOLESKY_PARENT.__code__.co_code == old._slr_virtual_rotation_envelope.__code__.co_code
        assert current._O21_SLR_CHOLESKY_PARENT.__code__.co_consts == old._slr_virtual_rotation_envelope.__code__.co_consts
        assert not {'bool','all','float','int','item','tolist'}.intersection(current._slr_virtual_rotation_envelope.__code__.co_names)
        assert current._slr_finish_factors.__globals__['_slr_virtual_rotation_envelope'] is current._slr_virtual_rotation_envelope
        assert current._slr_inner.__globals__['_slr_virtual_rotation_envelope'] is current._slr_virtual_rotation_envelope
        assert current._astra_fhm_core.__globals__['_dcr_right'] is current._dcr_right
        assert 'bool' not in current._dcr_right.__code__.co_names and 'all' not in current._dcr_right.__code__.co_names
        assert '_ASTRA_BCC_PARENT_RIGHT' not in current._dcr_right.__code__.co_names
        assert current._K3LB_PARENT_LEFT is current._o20_npx_copy_exact_left_v1
        assert current._O20_NPX_COPY_EXACT_PARENT is current._astra_compute_left_v1
        assert current._ASTRA_COMPUTE_ON is False and current._ASTRA_COMPUTE_PARENT_V1 is current._npx_left
        assert current._np_left.__globals__['_K3LB_PARENT_LEFT'] is current._K3LB_PARENT_LEFT
        assert current._NP_K3L_BY_LAYER == old._NP_K3L_BY_LAYER and current._NP_STRENGTH == old._NP_STRENGTH
        assert not current._NP_CACHE and current._K3LB_LAYER[0] is None
        assert current._O20_SMDIRECT_PARENT.__name__ == '_astra_sm40_linear_grouped'
        assert current._GNN_PARAMS is None and current._EDGE_PARAMETERS is None and current._MESSAGE_EDGE_PARAMETERS is None
        assert not current._E2_GNN_STATE and not current._E2_EDGE_STATE
        for name in ('_GNN_LITERAL', '_EDGE_LITERAL', '_MESSAGE_EDGE_LITERAL'):
            assert getattr(current, name) == getattr(old, name), name
        assert current._VMFR_CM == old._VMFR_CM == 8 and current._O20_EDGE_HIDDEN == old._O20_EDGE_HIDDEN == 8
        assert current._O20_FLATEXTRA_ON is False
        assert current._hzd_load is current._O20_FLATEXTRA_HZD_PARENT
        assert current._rs_load_b_owned_ql_v1 is current._O20_FLATEXTRA_OWNED_PARENT
        assert all(current.__dict__.get(name) is value for name, value in current._ASTRA_DENSE_CONTEXT)
        assert current._nonlinear_pair_batch.__wrapped__ is current.nonlinear
        assert current._nonlinear_pair_batch.__qualname__ == '_tr_clearing.<locals>.w'
        assert current._nonlinear_pair_batch.__closure__[0].cell_contents is current.nonlinear
        assert current.features is current._re22_features
        assert current._RE22_PARENT_FEATURES.__name__ == 'features'
        assert current._fh90_first_head.__code__.co_code == old._fh90_first_head.__code__.co_code
        assert current._fh90_first_head.__code__.co_consts == old._fh90_first_head.__code__.co_consts
        assert current._fh90_first_head.__code__.co_names == old._fh90_first_head.__code__.co_names
        assert current._gop2_gnn_step.__code__.co_code == old._gop2_gnn_step.__code__.co_code
        assert current._gop2_gnn_step.__code__.co_consts == old._gop2_gnn_step.__code__.co_consts
        assert current._gop2_gnn_step.__code__.co_names == old._gop2_gnn_step.__code__.co_names
        plan = []; inputs = []; out = []
        assert not current._o20_smdirect_eligible(plan, out, None)
        assert current._sm40_program_key(plan, inputs, out, None) == current._O20_SMDIRECT_KEY_PARENT(plan, inputs, out, None) == (id(plan), id(inputs), id(out))
        assert current._O20_SMDIRECT_STATS == old._O20_SMDIRECT_STATS
        cases.append(dict(key=key, role=role, sha256=sha(path), parent_sha256=sha(parent),
                          parent_cache_key=list(parent_key), unchanged_model_literals=True,
                          live_aliases=True, live_physics_bytecode=True, grouped_key=True,
                          empty_caches=True, permanent_empty_rescue_map=True))
assert len(parents) == 2
report = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
              successful=True, numpy_free=True, no_setup=True, no_prediction=True,
              parent_cache_key='(parent_sha256, role)', distinct_parent_cache_entries=len(parents),
              cases=cases, script_sha256=sha(__file__), build_sha256=sha(HERE / 'BUILD_v1.json'))
with (HERE / 'IMPORT_CHECK_v1.json').open('x') as stream:
    json.dump(report, stream, indent=2); stream.write('\n')
print(json.dumps(report))
