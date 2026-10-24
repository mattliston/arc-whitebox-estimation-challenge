"""Independent selected-path accounting; deliberately does not import engine."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())


def pin(p):
    p = Path(p); data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def linear(source, incoming):
    # Half-open row and column intervals, reduced to their known upper limits.
    nodes, total = [], 0
    for node in source['nodes']:
        op, args = node['op'], node['args']
        if op == 'input':
            current = incoming[node['literal']]
        elif op == 'zero':
            current = (0, 0)
        elif op in ('add', 'sub'):
            current = tuple(max(nodes[args[0]][j], nodes[args[1]][j]) for j in (0, 1))
            total += current[0] * current[1]
        else:
            assert op in ('scale', 'neg')
            current = nodes[args[0]] if op == 'neg' or Fraction(node['literal']) else (0, 0)
            total += current[0] * current[1]
        nodes.append(current)
    result, uses = [], Counter()
    for item in source['outputs']:
        ref, scale = item['ref'], Fraction(item['scale'])
        current = nodes[ref] if scale else (0, 0)
        if scale != 1 or source['nodes'][ref]['op'] == 'input' or uses[ref]:
            total += current[0] * current[1]
        uses[ref] += 1
        result.append(current)
    return result, total


def bounds(rows, cols, rp, cp):
    h = -(-rows // rp); w = -(-cols // cp)
    result = []
    for index in range(rp * cp):
        y, x = divmod(index, cp)
        height = len(range(y * h, min((y + 1) * h, rows)))
        width = len(range(x * w, min((x + 1) * w, cols)))
        result.append((height, width) if height and width else (0, 0))
    return result


def main():
    seal = read(HERE / 'inputs_sealed.json')
    for row in seal['pins']:
        assert pin(row['path']) == row
    result = read(HERE / 'result.json')
    assert result['completed'] and result['source_seal'] == pin(HERE / 'inputs_sealed.json')
    sources = read(HERE / 'programs.json')
    graph = {tuple(map(int, k.split(','))): v for k, v in result['selected_graph'].items()}
    rebuilt = {}
    program_counts, zero_products = Counter(), 0
    for shape in sorted(graph, key=lambda s: (sum(s), s)):
        m, k, n = shape
        row = graph[shape]
        kind = row['kind']
        if kind == 'gemm':
            cost, memory = m * n * (2 * k - 1), 4 * (m * k + k * n + m * n)
        else:
            p = sources[row['program']]
            a, b, c = p['shape']; rank = p['rank']
            program_counts[p['name'] + ':' + kind] += 1
            if kind == 'padded_support':
                L, lc = linear(p['circuits']['L'], bounds(m, k, a, b))
                R, rc = linear(p['circuits']['R'], bounds(k, n, b, c))
                children, product_extents = Counter(), []
                for left, right in zip(L, R):
                    s = (left[0], min(left[1], right[0]), right[1])
                    if all(s):
                        children[s] += 1; product_extents.append((s[0], s[2]))
                    else:
                        product_extents.append((0, 0))
                _, pc = linear(p['circuits']['P'], product_extents)
                assert sorted(children.items()) == [(tuple(s), count) for s, count in row['children']]
                assert rank - sum(children.values()) == row['zero_products']
                zero_products += row['zero_products']
                assert row['linear'] == lc + rc + pc
                cost = lc + rc + pc + sum(count * rebuilt[s][0] for s, count in children.items())
                memory = sum(count * rebuilt[s][1] for s, count in children.items())
            elif kind == 'uniform_padding':
                child = (-(-m // a), -(-k // b), -(-n // c))
                assert child == tuple(row['child'])
                x, y, z = child; ca, cb, cc = p['packed_linear_operations']
                linear_cost = ca * x * y + cb * y * z + cc * x * z
                assert linear_cost == row['linear']
                cost = linear_cost + rank * rebuilt[child][0]
                memory = rank * rebuilt[child][1]
            else:
                assert kind == 'recursive_peel'
                child = (m // a, k // b, n // c)
                assert min(child) > 0 and child == tuple(row['core'])
                x, y, z = child; ca, cb, cc = p['packed_linear_operations']
                m0, k0, n0 = x * a, y * b, z * c
                fringes = []
                if k > k0: fringes.append((m0, k - k0, n0))
                if n > n0: fringes.append((m0, k, n - n0))
                if m > m0: fringes.append((m - m0, k, n))
                assert fringes and fringes == [tuple(s) for s in row['fringes']]
                linear_cost = ca * x * y + cb * y * z + cc * x * z + (m0 * n0 if k > k0 else 0)
                assert row['linear'] == linear_cost
                cost = linear_cost + rank * rebuilt[child][0] + sum(rebuilt[s][0] for s in fringes)
                memory = max([rank * rebuilt[child][1]] + [rebuilt[s][1] for s in fringes])
        assert (cost, memory) == (row['cost'], row['leaf_bytes']), shape
        rebuilt[shape] = (cost, memory)
    old = read(seal['prior_result'])
    inputs = read(seal['census_inputs'])
    gates, calls_checked = [], 0
    for row, census in zip(result['records'], inputs['censuses']):
        actual = read(census['path'])
        assert row['rank'] == census['rank']
        assert len(row['details']) == len(actual['outermost_mm_calls']) == 295
        savings = 0
        for detail, call in zip(row['details'], actual['outermost_mm_calls']):
            shape = (call['left'][0], call['left'][1], call['right'][1])
            assert shape == tuple(detail['shape'])
            cost, mem = rebuilt[shape] if min(shape) else (call['flops'], 0)
            assert cost == detail['optimistic_cost'] and mem == detail['selected_leaf_bytes']
            saved = max(0, call['flops'] - cost)
            assert saved == detail['optimistic_saved']
            savings += saved; calls_checked += 1
        total = actual['total_flops']
        extended = next(r for r in old['families'][1]['records'] if r['rank'] == row['rank'])
        additional = savings - extended['optimistic_saved']
        gate = 100 * additional >= 3 * total
        assert gate == row['passes_three_percent_incremental_gate']
        assert savings == row['optimistic_saved'] and total - savings == row['optimistic_new_total']
        assert additional == row['incremental_saved_vs_extended']
        gates.append(gate)
    assert all(gates) == result['opportunity_gate_passed']
    receipt = dict(passed=True, selected_nodes_replayed=len(rebuilt),
                   census_calls_replayed=calls_checked, opportunity_gate_passed=all(gates),
                   selected_graph_program_counts=dict(program_counts),
                   selected_graph_zero_products_skipped=zero_products,
                   result=pin(HERE / 'result.json'), source=pin(Path(__file__).resolve()),
                   scope='Independent selected arithmetic/leaf-storage/census replay; not an independent optimality proof, native bill or runtime/accuracy qualification.')
    with (HERE / 'replay.json').open('x') as out:
        json.dump(receipt, out, indent=2); out.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
