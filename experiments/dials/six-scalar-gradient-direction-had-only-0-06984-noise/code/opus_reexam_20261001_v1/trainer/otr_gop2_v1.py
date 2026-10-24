"""CPU mixed-precision GOP2 forward; detached V/Q training approximation.

No tensor/model imports or numerical work at module entry. All per-call bases
are recomputed, never cached. OFF is implemented by not installing this adapter.
"""
import ast
import copy
import hashlib
from pathlib import Path

SOURCE_SHA = '63bb5d15786995b74d761a6f6690170710ab397497351b67f0cf794401362f94'

def contract(T, X, torch):
    """Unbatched equivalent of native _gop2_contract (one of its batch items).

Forward exactly follows the native precision/order specification. V and Q are
detached bases, as explicitly authorized for training; this is not a claim of
the full derivative through a spectral truncation or rank-deficient QR.
    """
    assert T.device.type == X.device.type == 'cpu', 'GOP2 requires CPU float64 spectral work; MPS substitution is forbidden'
    assert T.dtype == X.dtype == torch.float32
    assert T.ndim == X.ndim == 2 and T.shape[1] == X.shape[0] and X.shape[1] == 16
    mu = X.mean(dim=0)
    centered = X - mu[None, :]
    with torch.no_grad():
        wide = centered.detach().to(torch.float64)
        gram = wide.T @ wide
        _, vectors = torch.linalg.eigh(gram)
        V = vectors[:, -2:].to(torch.float32)
        coefficients = centered.detach() @ V
        projected = T.detach() @ coefficients
        Q, _ = torch.linalg.qr(projected.to(torch.float64), mode='reduced')
        Q = Q.to(torch.float32)
    back = Q.T @ T
    fitted = back @ centered
    residual = Q @ fitted
    mass = T.sum(dim=1)
    return residual + mass[:, None] * mu[None, :]

def patch_ast(node):
    """Only replace the designated even-power contractions; retain odd ASTs."""
    assert node.name in ('gnn_step', '_message_edge_source', '_message_edge_source_ein')
    targets = {'m2', 'e1', 'e2', 'e3'} if node.name == 'gnn_step' else {'m2'}
    found = []
    for item in ast.walk(node):
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name) and item.targets[0].id in targets:
            assert isinstance(item.value, ast.BinOp) and isinstance(item.value.op, ast.MatMult)
            item.value = ast.Call(func=ast.Name(id='_astra_gop2_contract', ctx=ast.Load()), args=[item.value.left, item.value.right], keywords=[])
            found.append(item.targets[0].id)
    assert set(found) == targets and len(found) == len(targets)
    return node

def install(cd, torch):
    import model_bank_bfix as MB
    assert cd.w.device.type == 'cpu', 'GOP2 carry is CPU-only'
    source = Path(MB.SOURCE)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    tree = ast.parse(source.read_text())
    cd.env['_astra_gop2_contract'] = lambda T, X: contract(T, X, torch)
    for name in ('gnn_step', '_message_edge_source'):
        original = cd.env[name]
        assert original.__code__.co_name == name
        assert original.__code__.co_filename == str(source) + ':explicit_offline_translation'
        node = copy.deepcopy(next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name))
        modified = patch_ast(node)
        translated = ast.fix_missing_locations(MB.Translate().visit(ast.Module(body=[modified], type_ignores=[])))
        # Preserve the card environment and the existing wrapper provenance
        # contract. The manifest separately pins this adapter and changed AST.
        exec(compile(translated, original.__code__.co_filename, 'exec'), cd.env)
