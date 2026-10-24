"""Exclusive source/metadata-only build. Never imports or executes a model."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
PARENT = H2 / 'hill_climb_star_projection32_v1/candidate.py'
PARENT_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
CENSUS = H2 / 'hill_climb_s32_bill_census_v1/census.json'


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    parent = PARENT.read_bytes()
    fragment = (HERE / 'fragment.py').read_bytes()
    tree = ast.parse(parent)
    old = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_PB_SINGLE' for t in node.targets):
            try:
                old = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    addition = ast.parse(fragment).body[0]
    assert isinstance(addition, ast.Assign) and isinstance(addition.value, ast.BinOp)
    keys = ast.literal_eval(addition.value.left)
    assert len(old) == len(set(old)) == 600
    assert len(keys) == len(set(keys)) == 27
    assert not set(keys).intersection(old)
    assert len(old + keys) == len(set(old + keys)) == 627
    counts = Counter(k[0] for k in keys + old)
    caps = {0: 320, 1: 352, 2: 320}
    assert all(counts[k] < caps[k] for k in caps)

    census = json.loads(CENSUS.read_text())
    assert census['passed'] and census['candidate']['sha256'] == PARENT_SHA
    assert census['bill'] == census['attributed'] == 276549645399
    assert census['all16_bytes_exact'] and census['full_bill_exact'] and census['hook_restored']
    observed = set()
    for row in census['rows']:
        if row['op'] not in ('matmul', 'einsum') or row['inner'][0] not in ('_ipt_leaf_unwind_ab', '_ks_product'):
            continue
        if len(row['shapes']) < 2:
            continue
        a, b = row['shapes'][:2]
        if len(a) < 4 or a[:-2] != b[:-2] or any(v != 7 for v in a[:-2]):
            continue
        depth = len(a) - 2
        u = 2 ** depth
        m, k, n = a[-2] * u, a[-1] * u, b[-1] * u
        observed.update(((0, depth, m, k), (1, depth, k, n), (2, depth, m, n)))
    assert set(keys).issubset(observed)
    rows = []
    for kind, depth, r, c in keys:
        u = 2 ** depth
        assert r % u == c % u == 0
        tail = (r // u) * (c // u)
        tree_flops = 7 ** depth * tail
        level1_flops = 4 ** (depth - 1) * 7 * tail
        rows.append(dict(key=[kind, depth, r, c], tree_reshape_flops=tree_flops,
                         level1_reshape_flops=level1_flops,
                         predicted_first_bind_flops=tree_flops + level1_flops))
    assert sum(r['tree_reshape_flops'] for r in rows) == 1384690314
    candidate = parent + fragment
    ast.parse(candidate)
    assert candidate[:len(parent)] == parent and candidate[len(parent):] == fragment
    with (HERE / 'candidate.py').open('xb') as f:
        f.write(candidate)
    primary = sum(r['tree_reshape_flops'] for r in rows)
    secondary = sum(r['level1_reshape_flops'] for r in rows)
    report = dict(
        source_only=True, numerical_execution=False,
        parent=pin(PARENT), candidate=pin(HERE / 'candidate.py'),
        fragment=pin(HERE / 'fragment.py'), builder=pin(Path(__file__)),
        protocol=pin(HERE / 'PROTOCOL.md'),
        sources=[pin(CENSUS), pin(CENSUS.with_name('summary.json')), pin(CENSUS.with_name('guard.json'))],
        inverse_exact=True, exact_parent_prefix=True, appended_bytes=len(fragment),
        keys=keys, no_duplicate_entries=True, old_keys=600, new_keys=627,
        setup_expected=dict(parent_bound=624, parent_fused=600, candidate_bound=651, candidate_fused=627,
                            with_d3_bound=660, with_d3_fused=636,
                            condition='Fresh setup completes both phases before unchanged four-second guard.'),
        kind_counts={str(k): counts[k] for k in caps}, cache_caps={str(k): caps[k] for k in caps},
        pricing=dict(rows=rows, tree_reshape_flops=primary, level1_reshape_flops=secondary,
                     predicted_rank35_first_bind_reduction=primary + secondary,
                     fraction_of_rank35_bill=(primary + secondary) / census['bill'],
                     predicted_rank35_bill_without_eviction=census['bill'] - primary - secondary,
                     warm_reduction_if_all_entries_already_resident=0,
                     caveat='Pricing is setup transfer of first binds, not a steady per-network reduction or worker-average claim.'),
        composition_sources=[pin(H2 / 'hill_climb_star32_fused_d3_v1/candidate.py'),
                             pin(H2 / 'hill_climb_uvp_base_bcache_v1/candidate.py')])
    with (HERE / 'build.json').open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('candidate', 'setup_expected', 'kind_counts')}))
    print(json.dumps({k: v for k, v in report['pricing'].items() if k != 'rows'}))


if __name__ == '__main__':
    main()
