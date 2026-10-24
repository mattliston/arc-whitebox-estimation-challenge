"""Batch the already verified restricted HK coefficient expressions.

Source formulas are local and pinned. No model/target data are read here.
"""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
old = HERE.parent / 'codex_resume_20260926_v1/hk233_fragment_v1.py'
text = old.read_text()
tree = ast.parse(text)
f = tree.body[0]
assert isinstance(f, ast.FunctionDef) and f.name == '_hk233_product'
f.name = '_hkb233_product'
f.args.args[2].arg = 'mm_many'
pairs, names, body = [], [], []
inserted = False
for node in f.body:
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith('m_')):
        call = node.value
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        assert call.func.attr == 'copy' and not call.args
        product = call.func.value
        assert isinstance(product, ast.Call) and isinstance(product.func, ast.Name)
        assert product.func.id == 'mm' and len(product.args) == 2
        pairs.append(ast.Tuple(elts=product.args, ctx=ast.Load()))
        names.append(node.targets[0])
        continue
    if pairs and not inserted:
        assert len(pairs) == 15
        body.append(ast.Assign(targets=[ast.Tuple(elts=names, ctx=ast.Store())],
            value=ast.Call(func=ast.Name(id='mm_many', ctx=ast.Load()),
                           args=[ast.List(elts=pairs, ctx=ast.Load())], keywords=[])))
        inserted = True
    body.append(node)
assert inserted and len(pairs) == 15
f.body = body
ast.fix_missing_locations(tree)
generated = ast.unparse(tree) + '\n\n' + '''
def _hkb6_many(pairs):
    """Largest complete batch that fits the unchanged depth-six arena."""
    m, k, n = int(pairs[0][0].shape[0]), int(pairs[0][0].shape[1]), int(pairs[0][1].shape[1])
    assert m % 64 == k % 64 == n % 64 == 0
    assert all(tuple(a.shape) == (m, k) and tuple(b.shape) == (k, n)
               and str(a.dtype) == str(b.dtype) == 'float32' for a, b in pairs)
    counts = (7**6 * (m//64) * (k//64),
              7**6 * (k//64) * (n//64),
              7**6 * (m//64) * (n//64))
    cap = min(15, *(r // q for r, q in zip(_IPT_REGION[6], counts)))
    assert cap >= 1 and _AB_ON and _SCRATCH is not None
    outs = [fnp.empty((m, n), dtype=fnp.float32) for _ in pairs]
    _TR_LAST[0] = None
    for start in range(0, len(pairs), cap):
        block = pairs[start:start + cap]
        target = outs[start:start + cap]
        _bb_product([a for a, b in block], [b for a, b in block], target, 6)
    _TR_LAST[0] = None
    return outs
'''
target = HERE / 'hk233_batch_fragment_v1.py'
compile(generated, str(target), 'exec')
with target.open('x') as stream:
    stream.write(generated)
record = dict(parent_helper=str(old), parent_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    status='Research helper; untested batching of identical algebra',
    batch_rule='max P <=15 fitting every unchanged depth6 arena region; no new arena')
with (HERE / 'hk233_batch_manifest_v1.json').open('x') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record))
