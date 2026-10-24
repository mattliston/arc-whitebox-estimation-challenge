"""Source-only, exclusive, exactly reversible S32 cache build. No model imports."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'hill_climb_star_projection32_v1/candidate.py'
PARENT_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'


def pin(p):
    b = p.read_bytes()
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    source = PARENT.read_text()
    original = source
    nodes = {(n.name, n.lineno): n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    edits = []
    requests = [
        ('_uvp_recon', 8063, [
            ('_shared_old384_mm(U[:, :ru], Fu)', "_ubc_product(U[:, :ru], Fu, uv, 'u')"),
            ('_shared_old384_mm(U, Fv2)', "_ubc_product(U, Fv2, uv, 'v')")]),
        ('_mm', 5313, [('b[:, j:min(j + _PACKED_TILE, n)]', '_ubc_slice(b, j, min(j + _PACKED_TILE, n))')]),
        ('_push5_product', 4208, [('b[:, :4096]', '_ubc_slice(b, 0, 4096)'), ('b[:, 4096:]', '_ubc_slice(b, 4096, n)')]),
        ('_cp_d6_shared', 6021, [('b[:, j0:j1]', '_ubc_slice(b, j0, j1)')]),
        ('_push6_padded', 4220, [
            ('    hold.append(fnp.copyto(bb[:k, :n], b))\n', '    hold.append(fnp.copyto(bb[:k, :n], b))\n    _ubc_padding(b, bb, k, n, kp, npad)\n')]),
        ('_ipt_product_ab', 7932, [
            ("            _P6_STATS['moved'] += 1\n", "            _ubc_padding(b, bb, k, n, K, N)\n            _P6_STATS['moved'] += 1\n")]),
    ]
    for name, line, replacements in requests:
        node = nodes[(name, line)]
        before = '\n'.join(original.splitlines()[node.lineno - 1:node.end_lineno])
        after = before
        for old, new in replacements:
            assert after.count(old) == 1, (name, old)
            after = after.replace(old, new)
        assert source.count(before) == 1, name
        source = source.replace(before, after)
        edits.append(dict(function=name, original_line=line, before=before, after=after))
    fragment = (HERE / 'fragment.py').read_text()
    candidate = source + '\n\n' + fragment
    ast.parse(candidate)
    inverse = source
    for edit in reversed(edits):
        assert inverse.count(edit['after']) == 1
        inverse = inverse.replace(edit['after'], edit['before'])
    assert inverse == original
    with (HERE / 'candidate.py').open('x') as f:
        f.write(candidate)
    report = dict(parent=pin(PARENT), candidate=pin(HERE / 'candidate.py'),
                  fragment=pin(HERE / 'fragment.py'), build=pin(Path(__file__)),
                  protocol=pin(HERE / 'PROTOCOL.md'), edits=edits,
                  inverse_exact=True, numerical_execution=False)
    with (HERE / 'build.json').open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
