"""Pure shape/cost planning; no network weights, predictions, or labels."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FABLE = HERE.parent / 'fable_wall_repair_v1'


def kernel_bill(m, k, n, depth):
    if depth == 6 and n > 4096:
        return kernel_bill(m, k, 4096, depth) + kernel_bill(m, k, n - 4096, depth)
    growth = (7 / 4) ** depth
    return round(2 * m * k * n * (7 / 8) ** depth - m * n * growth
                 + (7 / 3) * (m * k + k * n + m * n) * (growth - 1))


def requests(n, depth):
    # Warm kernel: 18 arithmetic/copy calls per level, one leaf, 12 corner
    # slices, and one output allocation. D6 wide pushes make two packets.
    if depth == 0:
        return 1
    return (18 * depth + 14) * (2 if depth == 6 and n > 4096 else 1)


def main():
    bench = json.loads((FABLE / 'bench_depths.json').read_text())
    census = json.loads((FABLE / 'census_variantE_contig_cached_r35.json').read_text())
    source = (FABLE / 'cand_c2_wallrepair_I95.py').read_text()
    tree = ast.parse(source)
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('_SQUARE_EQUAL_DEPTHS', '_SQUARE_CROSS_DEPTHS', '_SQUARE_COSTS'):
                    values[target.id] = ast.literal_eval(node.value)
    rows = []
    for c in census:
        m, k, n = (c[x] for x in ('m', 'k', 'n'))
        b = bench[f'{m},{k},{n}']
        for depth, v in b['depths'].items():
            assert kernel_bill(m, k, n, int(depth)) == v['bill']
        cap = 6 if b['kind'] == 'push' else 4
        if b['kind'] == 'square':
            if m == n:
                cap = values['_SQUARE_EQUAL_DEPTHS'][m // 32 - 1][k // 32 - 1]
            elif (m == k and n == 1024) or (k == n and m == 1024):
                cap = values['_SQUARE_CROSS_DEPTHS'][k // 32 - 1]
            else:
                cap = min((d for d in (0, 3, 4, 5) if str(d) in b['depths']),
                          key=lambda d: b['depths'][str(d)]['bill'])
            cap = min(cap, 5)
        choices = {int(d): v for d, v in b['depths'].items() if int(d) <= cap}
        rows.append(dict(m=m, k=k, n=n, calls=c['calls'], kind=b['kind'], choices=choices))
    out = []
    for penalty in (0, 250000, 500000, 750000, 1000000, 1250000, 1500000, 2000000, 2500000, 3000000, 4000000):
        cost = count = wall = 0
        decisions = {}
        for r in rows:
            d = min(r['choices'], key=lambda d: r['choices'][d]['bill'] + penalty * requests(r['n'], d))
            rec = r['choices'][d]
            cost += r['calls'] * rec['bill']
            count += r['calls'] * requests(r['n'], d)
            wall += r['calls'] * rec['warm_s']
            decisions[f"{r['m']},{r['k']},{r['n']}"] = d
        out.append(dict(penalty=penalty, kernel_bill=cost, warm_kernel_requests=count,
                        saved_boundary_kernel_s=wall, decisions=decisions))
    with (HERE / 'cost_rule_plan.json').open('x') as f:
        json.dump(dict(benchmark_cost_formula_exact=True, plans=out), f, indent=2)
    for r in out:
        print({k: v for k, v in r.items() if k != 'decisions'})


if __name__ == '__main__':
    main()
