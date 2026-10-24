"""Target-free observer lifecycle proofs; fake arrays, no model or truth execution."""
import ast
import gc
import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/arcworker/arc-workspace')
SCI = HERE.parent/'runtime_first128_frozen_census_v1/candidate_first128_paid_nocensus_science_v1.py'
SCI_SHA = '49bd2ba31f0bce983f44cc001746e63ffaa84e530d3b31efa6424f69a0298c80'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Array:
    def __init__(self, shape, value=0.0, dtype='float32'):
        self.shape, self.value, self.dtype = shape, value, dtype
    def tolist(self):
        if len(self.shape) == 1:
            return [self.value]*self.shape[0]
        return [[self.value]*self.shape[1] for _ in range(self.shape[0])]
    def __getitem__(self, key):
        assert key == (slice(None), 5)
        return Array((1024,), self.value)

MOCK = '''
def _fh90_first_head(k, l):
    if k.get('raise_first'):
        raise RuntimeError('synthetic first helper error')
    return 1

def _gop2_gnn_step(l, x, sigma, W, state, P):
    if l < 1:
        return sigma, state
    k = P
    h = P['first']
    scalar = _fh90_first_head(k, l)
    if P.get('raise_after_first'):
        raise RuntimeError('synthetic step error')
    h = P['hidden']
    if P.get('omit_delta'):
        return sigma, state
    delta = P['delta']
    return delta, state
'''

def fixture():
    mod = types.ModuleType('op1_mock_source')
    exec(compile(MOCK, '<op1_mock_source>', 'exec'), mod.__dict__)
    mod.__file__ = str(SCI)
    mod._GOP2_ON, mod._FAST_RESCUE_FRAC, mod._GNN_LITERAL = True, {}, {}
    mod._hc90_hidden_head = mod._E2Hold = mod._hc_bias = lambda *args: None
    teacher = dict(path=str(SCI.relative_to(ROOT)), sha256=SCI_SHA, effective_sha256=SCI_SHA)
    recipe = dict(teacher=teacher, target_free=True, uses_network_truth=False, rows_per_layer=1024,
                  layers=list(range(1, 16)), reference_submission=337466, capture_ranks=[7100120],
                  source_family=dict(science=teacher, train=[7100120], validation=[7100279]),
                  source_manifest={}, on_policy_prereg={}, original_group_indices=[0])
    return mod, recipe

def inputs():
    return Array((1024,), 0.3), Array((1024, 64), 0.4), dict(
        first=Array((1024, 128), 0.125), hidden=Array((1024, 128), 0.25),
        delta=Array((1024,), 0.5), H=Array((1024, 78), 0.75))

def main():
    assert sys.platform == 'linux' and sha(SCI) == SCI_SHA
    hook_path = HERE/'first128_onpolicy_hook_v2.py'
    spec = importlib.util.spec_from_file_location('op1_synthetic_observer', hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    original_profile, original_gc = sys.getprofile(), (gc.isenabled(), gc.get_threshold())
    forwarded = [0]
    def prior(frame, event, arg):
        forwarded[0] += 1
    proofs = {}
    try:
        sys.setprofile(prior)
        mod, recipe = fixture()
        aliases = {name: id(getattr(mod, name)) for name in ('_gop2_gnn_step', '_fh90_first_head', '_hc90_hidden_head', '_E2Hold', '_hc_bias')}
        with tempfile.TemporaryDirectory(prefix='arc_op1_observer_success_') as tmp:
            observer = hook.install(mod, tmp, recipe)
            for phase in ('warmup', 'scored'):
                observer.begin(7100120, phase)
                sigma, state, P = inputs()
                assert mod._gop2_gnn_step(0, None, sigma, None, state, P) == (sigma, state)
                for layer in range(1, 16):
                    mod._gop2_gnn_step(layer, None, sigma, None, state, P)
                    assert not observer.pending
            result = observer.finish()
            assert sys.getprofile() is prior and forwarded[0] > 0
            assert len(result['files']) == result['direct_first128_count'] == 30
            assert result['valid_layer0_returns_skipped'] == 2
            assert result['all_first_paired_once'] and not result['native_handles_or_frames_retained']
            assert result['native_handles_retained'] is False and not observer.pending
            for item in result['files']:
                with gzip.open(Path(tmp)/'gnn_full_state_v1'/item['path'], 'rt') as stream:
                    data = json.load(stream)
                assert data['first128']['shape'] == data['hidden2']['shape'] == [1024, 128]
                assert data['first128']['data'][0][0] == 0.125 and data['hidden2']['data'][0][0] == 0.25
                assert data['gate_feature']['data'][0] == 0.75
            assert aliases == {name: id(getattr(mod, name)) for name in aliases}
            assert (gc.isenabled(), gc.get_threshold()) == original_gc
            observer.abort('successful client cleanup is a no-op')
            assert sys.getprofile() is prior
        proofs.update(exact_first_caller_before_overwrite=True, all30_own_records_paired=True,
                      valid_layer0_early_returns_no_feature_fetch=True, profile_chain_forwarded_restored=True,
                      no_pending_native_handle_or_frame=True, original_aliases_gc_unchanged=True)
        cases = ('duplicate', 'wrong_caller', 'negative_layer', 'layer16', 'wrong_first_shape',
                 'wrong_first_dtype', 'wrong_state_shape', 'missing_delta', 'helper_exception',
                 'step_exception', 'bad_phase', 'unfinished_group', 'client_exception')
        for case in cases:
            mod, recipe = fixture()
            with tempfile.TemporaryDirectory(prefix='arc_op1_observer_failure_') as tmp:
                observer = hook.install(mod, tmp, recipe)
                sigma, state, P = inputs()
                failed = False
                try:
                    observer.begin(7100120, 'invalid' if case == 'bad_phase' else 'scored')
                    if case == 'wrong_first_shape': P['first'] = Array((1024, 127))
                    if case == 'wrong_first_dtype': P['first'] = Array((1024, 128), dtype='float64')
                    if case == 'wrong_state_shape': state = Array((1024, 63))
                    if case == 'missing_delta': P['omit_delta'] = True
                    if case == 'helper_exception': P['raise_first'] = True
                    if case == 'step_exception': P['raise_after_first'] = True
                    if case == 'wrong_caller':
                        mod._fh90_first_head(P, 1)
                    elif case == 'client_exception':
                        try:
                            raise RuntimeError('synthetic client failure before prediction')
                        finally:
                            observer.abort('client exit before successful finish')
                    else:
                        layer = -1 if case == 'negative_layer' else 16 if case == 'layer16' else 1
                        mod._gop2_gnn_step(layer, None, sigma, None, state, P)
                        if case == 'duplicate': mod._gop2_gnn_step(layer, None, sigma, None, state, P)
                        observer.finish()
                except (AssertionError, RuntimeError, KeyError):
                    failed = True
                finally:
                    observer.abort('synthetic outer client finally')
                assert failed, case
                assert sys.getprofile() is prior and not observer.pending, case
                assert not (Path(tmp)/'gnn_full_state_v1/CAPTURE_COMPLETE_v1.json').exists(), case
                proofs[case+'_fails_and_restores_without_success_receipt'] = True
        # Effective production wrapper bypasses the observed step for layer zero.
        tree = ast.parse(SCI.read_text())
        wrapper = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gnn_step'][-1]
        first_if = wrapper.body[0]
        assert isinstance(first_if, ast.If) and ast.unparse(first_if.test) == 'not _GOP2_ON or l < 1'
        assert isinstance(first_if.body[0], ast.Return)
        assert first_if.body[0].value.func.id == '_GOP2_PARENT_GNN'
        client = ast.parse((HERE/'client_capture_v1.py').read_text())
        outer = client.body[-1]
        assert isinstance(outer, ast.Try) and outer.finalbody
        assert 'observer.abort' in ast.unparse(outer.finalbody[-1])
        proofs['production_wrapper_bypasses_exact_observed_step_for_l0'] = True
        proofs['capture_client_outer_finally_aborts_on_all_exit_paths'] = True
    finally:
        sys.setprofile(original_profile)
    assert sys.getprofile() is original_profile and (gc.isenabled(), gc.get_threshold()) == original_gc
    result = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
                  successful=True, proofs=proofs, script_sha256=sha(__file__), observer_sha256=sha(hook_path),
                  client_sha256=sha(HERE/'client_capture_v1.py'), source_sha256=SCI_SHA,
                  no_truth_arrays_or_native_model_execution=True, native_anchor_parity_still_required=True)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
