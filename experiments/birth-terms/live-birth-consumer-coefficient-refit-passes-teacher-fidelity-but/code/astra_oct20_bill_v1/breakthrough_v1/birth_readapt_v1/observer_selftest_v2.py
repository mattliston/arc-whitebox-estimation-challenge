"""Check actual14-layer observer pattern before expensive native admission."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import types
import birth_capture_hook_v1 as hook

HERE = Path(__file__).resolve().parent
source = HERE.parents[2] / 'fable_oct19_kc_v1/candidate_micro_copyall_ecut_eh14_astra_sci_v1.py'
tree = ast.parse(source.read_text())
A = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == '_BDNH_A_LITERAL' for t in n.targets))


class Fake:
    def __init__(self, shape):
        self.shape = shape

    def _fetch_data(self):
        count = 1
        for n in self.shape:
            count *= n
        return bytes(count * 4), self.shape, 'float32'


namespace = {'Fake': Fake}
code = '\n' * 2374 + '''def _e2_bdnh_coefficients(H, A, cstar_all, l):
    standardized = H
    response = Fake((1024, 4))
    cstar = Fake((4,))
    onem = Fake((4,))
    coefficients = Fake((4, 1024))
    return coefficients
'''
exec(compile(code, 'birth_observer_mock', 'exec'), namespace)
mod = types.SimpleNamespace(__file__=str(source), _FAST_RESCUE_FRAC={},
                            _BDNH_A_LITERAL=A, _GNN_LITERAL={'msg_w': ((48, 64), None)},
                            _e2_bdnh_coefficients=namespace['_e2_bdnh_coefficients'])
before = sys.getprofile()
with tempfile.TemporaryDirectory(prefix='arc_birth_hook_') as folder:
    observer = hook.install(mod, folder, {'teacher': {'effective_sha256': hook.SOURCE}})
    for rank in (35, 150):
        observer.begin(rank, 'scored')
        for layer in range(1, 15):
            value = mod._e2_bdnh_coefficients(Fake((1024, 64)), None, None, layer)
            assert value.shape == (4, 1024)
    receipt = observer.finish()
    assert len(receipt['files']) == 28 and receipt['ranks'] == [35, 150]
    assert sys.getprofile() is before
result = dict(utc=hook.utc(), status='PASS_MOCK_ACTUAL_14_LAYER_TWO_RANK_CALL_PATTERN',
              hook_sha256=hook.sha(HERE / 'birth_capture_hook_v1.py'), source_sha256=hook.sha(source),
              records=28, all1024_rows=True, profile_restored=True, returned_value_preserved=True,
              truth_access=False, native_predictor_launches=0)
hook.write(HERE / 'OBSERVER_SELFTEST_v2.json', result)
print(json.dumps(result))
