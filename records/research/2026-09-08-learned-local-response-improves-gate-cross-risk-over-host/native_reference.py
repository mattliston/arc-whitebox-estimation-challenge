"""Inert AST-only extraction of the pinned current native local functions.

The original estimator module, embedded payload and predict are never imported.
The optional uncast-feature helper removes only the final cast, preserving its
upstream native arithmetic. No array is created merely by importing this file.
"""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BANK = HERE.parent/'unused_materializations_v1/estimator.py'
BANK_SHA = '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'


def original_functions(np):
    """Return ordinary NumPy equivalents, retaining all native dtype choices."""
    assert hashlib.sha256(BANK.read_bytes()).hexdigest() == BANK_SHA
    from flopscope.stats._norm import norm
    tree = ast.parse(BANK.read_text())
    names = {'gaussian_raw', 'derivatives', '_nonlinear_pair_batch',
             'features', 'gnn_step', 'silu'}
    chosen = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            assert node.name not in chosen, ('Duplicate native function', node.name)
            chosen[node.name] = node
    assert set(chosen) == names

    class RenameNumpy(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'fnp':
                return ast.copy_location(ast.Name(id='np', ctx=node.ctx), node)
            return node

    transformed = [RenameNumpy().visit(chosen[k]) for k in sorted(names)]
    uncast = copy.deepcopy(next(n for n in transformed if n.name == 'features'))
    uncast.name = 'features_uncast'
    returns = [n for n in ast.walk(uncast) if isinstance(n, ast.Return)]
    assert len(returns) == 1
    value = returns[0].value
    assert isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
    assert value.func.attr == 'astype' and len(value.args) == 1
    assert ast.dump(value.args[0]) == ast.dump(ast.parse('np.float32', mode='eval').body)
    returns[0].value = value.func.value
    module = ast.fix_missing_locations(ast.Module(body=transformed+[uncast], type_ignores=[]))
    env = dict(np=np, flops=SimpleNamespace(stats=SimpleNamespace(
        norm=SimpleNamespace(cdf=norm._compute_cdf))))
    exec(compile(module, str(BANK)+'::native_local_functions', 'exec'), env)
    return env
