"""Private current95c W3 forward adapter for worker18; never submission code.

Forward uses native float64 response-metric eigenspaces, float32 contractions.
For future training L/B are detached, like the existing GOP2 spectral adapter;
this is explicitly an approximate derivative, not full eigenvector backprop.
No tensor/model imports or execution at module import.
"""
import ast
import copy
from pathlib import Path


def contract(T, X, encoder, t):
    assert T.device.type == X.device.type == encoder.device.type == 'cpu'
    assert T.dtype == X.dtype == t.float32 and X.shape[1] == 16
    with t.no_grad():
        E = encoder.detach()[:, 62:78].to(t.float64)
        values, vectors = t.linalg.eigh(E.T @ E)
        scales = values.clamp_min(1e-30).sqrt()
        root = (vectors * scales[None, :]) @ vectors.T
        inverse = (vectors / scales[None, :]) @ vectors.T
        wide = X.detach().to(t.float64)
        gram = wide.T @ wide
        _, vectors = t.linalg.eigh((root @ gram) @ root)
        V = vectors[:, -4:]
        L = (root @ V).to(t.float32)
        B = (V.T @ inverse).to(t.float32)
    return (T @ (X @ L)) @ B


def patch_ast(node, parent_patch):
    """Retain all GOP2 edits and replace only the original odd W3 message."""
    node = parent_patch(node)
    found = 0
    for item in ast.walk(node):
        if (isinstance(item, ast.Assign) and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
                and item.targets[0].id == 'm3'):
            assert isinstance(item.value, ast.BinOp) and isinstance(item.value.op, ast.MatMult)
            item.value = ast.Call(func=ast.Name(id='_astra_current_w3_contract', ctx=ast.Load()),
                                  args=[item.value.left, item.value.right], keywords=[])
            found += 1
    assert found == 1, (node.name, found)
    return node


def install_worker_module(M, t):
    """Patch only this process's installer; do not reexecute M module globals."""
    import otr_gop2_v1 as G
    assert not getattr(M, '_current95c_w3_private', False), 'install once'
    original_patch = G.patch_ast

    def install_card(cd, torch):
        import hashlib
        import model_bank_bfix as MB
        source = Path(MB.SOURCE)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == G.SOURCE_SHA
        tree = ast.parse(source.read_text())
        cd.env['_astra_gop2_contract'] = lambda T, X: G.contract(T, X, torch)
        cd.env['_astra_current_w3_contract'] = lambda T, X: contract(T, X, cd.P['enc0_w'], torch)
        for name in ('gnn_step', '_message_edge_source'):
            original = cd.env[name]
            assert original.__code__.co_name == name
            assert original.__code__.co_filename == str(source) + ':explicit_offline_translation'
            node = copy.deepcopy(next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name))
            node = patch_ast(node, original_patch)
            translated = ast.fix_missing_locations(MB.Translate().visit(ast.Module(body=[node], type_ignores=[])))
            exec(compile(translated, original.__code__.co_filename, 'exec'), cd.env)

    # M later reconstructs an EDGEIN AST; it must use the same combined edits.
    M._install_gop2 = install_card
    M._gop2_patch_ast = lambda node: patch_ast(node, original_patch)
    M._current95c_w3_contract = contract
    source = Path(M.__file__).read_text()
    tree = ast.parse(source)
    installer = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'install')
    text = '\n'.join(source.splitlines()[installer.lineno-1:installer.end_lineno])
    before = "return t.cat((w.T @ src[:, :16], _gop2_contract(w2.T, src[:, 16:32], t), (w2 * w).T @ src[:, 32:]), 1).clamp(-8, 8).to(t.float64)"
    after = "return t.cat((w.T @ src[:, :16], _gop2_contract(w2.T, src[:, 16:32], t), _current95c_w3_contract((w2 * w).T, src[:, 32:], cd.P['enc0_w'], t)), 1).clamp(-8, 8).to(t.float64)"
    assert text.count(before) == 1
    text = text.replace(before, after)
    before = "                            m2 = _gop2_contract(w2.T, src[:, cm:2 * cm], t)"
    after = before + "\n                            m3 = _current95c_w3_contract(w3.T, src[:, 2 * cm:], P['enc0_w'], t)"
    assert text.count(before) == 1
    text = text.replace(before, after)
    import linecache
    filename = str(Path(__file__).resolve()) + ':private_M_install'
    linecache.cache[filename] = (len(text), None, text.splitlines(keepends=True), filename)
    exec(compile(text, filename, 'exec'), M.__dict__)
    M._current95c_w3_private = True
