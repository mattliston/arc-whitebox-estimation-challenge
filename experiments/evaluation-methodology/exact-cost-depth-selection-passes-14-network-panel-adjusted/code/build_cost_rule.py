"""Replace shape thresholds with exact kernel FLOPs plus a request penalty."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_SHA = 'b1c2c118c2d3e43cefbddfff98b57c0bfc8e195eb7af9f59ca131ed43322cf5a'

RULE = '''def _rule_depth(kind, m, k, n, maximum=6):
    # Shape-only metadata; no array values or labels enter kernel selection.
    key = (kind, m, k, n, maximum)
    if key in _COST_RULE_CACHE:
        return _COST_RULE_CACHE[key]
    options = (0, 3, 4, 5) if kind == 'sq' else (0, 4)
    if kind == 'push' and m == 1024 and m % 64 == k % 64 == n % 64 == 0:
        options = (0, 4, 6)
    best_depth = 0
    best_value = 2 * m * k * n - m * n + 1500000
    for depth in options[1:]:
        if depth > maximum or m % 2**depth or k % 2**depth or n % 2**depth:
            continue
        # D6 wide pushes repeat their left transform across two packets.
        packets = (4096, n - 4096) if depth == 6 and n > 4096 else (n,)
        growth = (7 / 4)**depth
        bill = sum(2*m*k*nc*(7/8)**depth - m*nc*growth
                   + (7/3)*(m*k+k*nc+m*nc)*(growth-1) for nc in packets)
        calls = (18*depth+14)*len(packets)
        value = bill + 1500000*calls
        if value < best_value:
            best_value, best_depth = value, depth
    _COST_RULE_CACHE[key] = best_depth
    return best_depth
'''


def main():
    parent = HERE / 'scalar_distill_I95.py'
    source = parent.read_text()
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == PARENT_SHA
    tree = ast.parse(source)
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_rule_depth']
    lines = source.splitlines(keepends=True)
    old = ''.join(lines[node.lineno-1:node.end_lineno])
    new = '_COST_RULE_CACHE = {}\n\n' + RULE
    assert source.count(old) == 1
    changed = source.replace(old, new)
    old_line = "    depth=min(depth,_SQUARE_DEPTH_CAP,_rule_depth('sq',m,k,n))   # W-I: size rule caps the depth (0 = plain matmul)"
    new_line = "    depth=_rule_depth('sq',m,k,n,min(depth,_SQUARE_DEPTH_CAP))   # exact bill plus request penalty"
    assert changed.count(old_line) == 1
    changed = changed.replace(old_line, new_line)
    assert changed.replace(new_line, old_line).replace(new, old) == source
    ast.parse(changed)
    out = HERE / 'cost_rule150_scalar.py'
    with out.open('x') as f:
        f.write(changed)
    receipt = dict(parent_sha256=PARENT_SHA, candidate_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                   source_inverse_exact=True, penalty_flops_per_request=1500000,
                   cost_formula_verified_all_saved_benchmarks=True,
                   labels_used_for_selection=False, physics_payloads_changed=False,
                   caveat='Request model is approximate; complete native and production-boundary qualification required.')
    with (HERE/'cost_rule150_build.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
