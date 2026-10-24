"""Pure-integer support geometry and exact scalar cost; no numerical imports."""
from collections import Counter
from fractions import Fraction
from functools import lru_cache


def winograd():
    def circuit(count, operations, outputs):
        nodes = [dict(op='input', args=[], literal=i) for i in range(count)]
        nodes += [dict(op=op, args=list(args), literal=None) for op, args in operations]
        return dict(inputs=count, nodes=nodes,
                    outputs=[dict(ref=i, scale='1') for i in outputs])
    left = circuit(4, [('add', (2, 3)), ('sub', (4, 0)),
                       ('sub', (0, 2)), ('sub', (1, 5))], [0, 1, 7, 3, 4, 5, 6])
    right = circuit(4, [('sub', (1, 0)), ('sub', (3, 4)),
                        ('sub', (3, 1)), ('sub', (5, 2))], [0, 2, 3, 7, 4, 5, 6])
    output = circuit(7, [('add', (0, 1)), ('add', (0, 5)), ('add', (8, 6)),
                         ('add', (8, 4)), ('add', (10, 2)), ('sub', (9, 3)),
                         ('add', (9, 4))], [7, 11, 12, 13])
    return dict(name='bank_winograd', shape=[2, 2, 2], rank=7,
                packed_linear_operations=[7, 7, 7],
                circuits=dict(L=left, R=right, P=output))


def compile_circuit(source):
    nodes = tuple((n['op'], tuple(n['args']),
                   Fraction(n['literal']) if n['op'] == 'scale' else n['literal'])
                  for n in source['nodes'])
    outputs = tuple((o['ref'], Fraction(o['scale'])) for o in source['outputs'])
    return source['inputs'], nodes, outputs


def shape2(rows, cols):
    return (rows, cols) if rows and cols else (0, 0)


def union(a, b):
    return max(a[0], b[0]), max(a[1], b[1])


def support(circuit, inputs, detail=False):
    count, nodes, outputs = circuit
    assert count == len(inputs)
    values, costs = [], []
    for i, (op, args, literal) in enumerate(nodes):
        assert all(0 <= j < i for j in args)
        if op == 'input':
            extent, cost = inputs[literal], 0
        elif op == 'zero':
            extent, cost = (0, 0), 0
        elif op in ('add', 'sub'):
            extent = union(values[args[0]], values[args[1]])
            cost = extent[0] * extent[1]
        elif op in ('neg', 'scale'):
            extent = values[args[0]] if op == 'neg' or literal else (0, 0)
            cost = extent[0] * extent[1]
        else:
            raise ValueError(op)
        values.append(extent)
        costs.append(cost)
    answer, output_costs, used = [], [], set()
    for ref, scale in outputs:
        extent = values[ref] if scale else (0, 0)
        area = extent[0] * extent[1]
        copy = nodes[ref][0] == 'input' or ref in used
        output_costs.append(area if scale != 1 or copy else 0)
        used.add(ref)
        answer.append(extent)
    total = sum(costs) + sum(output_costs)
    if detail:
        return tuple(answer), total, dict(nodes=values, node_costs=costs,
                                          output_costs=output_costs)
    return tuple(answer), total


def blocks(rows, cols, row_parts, col_parts):
    a, b = (rows + row_parts - 1) // row_parts, (cols + col_parts - 1) // col_parts
    return tuple(shape2(max(0, min(a, rows - i * a)),
                        max(0, min(b, cols - j * b)))
                 for i in range(row_parts) for j in range(col_parts))


def gemm(shape):
    m, k, n = shape
    assert min(shape) > 0
    return (2 * k - 1) * m * n


def leaf_bytes(shape):
    m, k, n = shape
    return 4 * (m * k + k * n + m * n)


class Geometry:
    def __init__(self, programs):
        self.programs = programs
        self.circuits = [dict((key, compile_circuit(c)) for key, c in p['circuits'].items())
                         for p in programs]

    @lru_cache(None)
    def side(self, index, which, rows, cols):
        p = self.programs[index]['shape']
        parts = p[:2] if which == 'L' else p[1:]
        return support(self.circuits[index][which], blocks(rows, cols, *parts))

    def padded(self, index, shape):
        m, k, n = shape
        left, lc = self.side(index, 'L', m, k)
        right, rc = self.side(index, 'R', k, n)
        children = []
        products = []
        for (a, b), (c, d) in zip(left, right):
            child = (a, min(b, c), d)
            if min(child):
                assert sum(child) < sum(shape), (index, shape, child)
                children.append(child)
                products.append((a, d))
            else:
                products.append((0, 0))
        outputs, pc = support(self.circuits[index]['P'], products)
        return lc + rc + pc, tuple(sorted(Counter(children).items())), len(left) - len(children)

    def peeled(self, index, shape):
        m, k, n = shape
        p, q, r = self.programs[index]['shape']
        a, b, c = m // p, k // q, n // r
        if min(a, b, c) == 0:
            return None
        m0, k0, n0 = a * p, b * q, c * r
        rm, rk, rn = m - m0, k - k0, n - n0
        if not (rm or rk or rn):
            return None
        ca, cb, cc = self.programs[index]['packed_linear_operations']
        linear = ca * a * b + cb * b * c + cc * a * c
        extra = m0 * n0 if rk else 0
        fringes = []
        if rk:
            fringes.append((m0, rk, n0))
        if rn:
            fringes.append((m0, k, rn))
        if rm:
            fringes.append((rm, k, n))
        assert all(sum(s) < sum(shape) for s in fringes)
        return linear + extra, (a, b, c), tuple(fringes)


class Search:
    def __init__(self, programs, maximum_shapes=250000):
        self.geometry = Geometry(programs)
        self.programs = programs
        self.memo = {}
        self.maximum_shapes = maximum_shapes
        self.maximum_sum_on_stack = 0

    def best(self, shape):
        shape = tuple(shape)
        if shape in self.memo:
            return self.memo[shape]
        assert len(self.memo) < self.maximum_shapes, 'Scalar cache shape limit'
        best = dict(cost=gemm(shape), leaf_bytes=leaf_bytes(shape), kind='gemm')
        if shape != (1, 1, 1):
            for index, p in enumerate(self.programs):
                linear, children, zeros = self.geometry.padded(index, shape)
                cost, memory = linear, 0
                for child, multiplicity in children:
                    row = self.best(child)
                    cost += multiplicity * row['cost']
                    memory += multiplicity * row['leaf_bytes']
                if (cost, memory) < (best['cost'], best['leaf_bytes']):
                    best = dict(cost=cost, leaf_bytes=memory, kind='padded_support',
                                program=index, linear=linear,
                                children=children, zero_products=zeros)
                # Explicitly retain the earlier uniform padded family. Known
                # zeros may always be ignored; no monotonicity assumption is
                # needed to assert inclusion of its complete recursive paths.
                child = tuple((v + part - 1) // part for v, part in zip(shape, p['shape']))
                a, b, c = child
                ca, cb, cc = p['packed_linear_operations']
                linear = ca * a * b + cb * b * c + cc * a * c
                row = self.best(child)
                cost = linear + p['rank'] * row['cost']
                memory = p['rank'] * row['leaf_bytes']
                if (cost, memory) < (best['cost'], best['leaf_bytes']):
                    best = dict(cost=cost, leaf_bytes=memory, kind='uniform_padding',
                                program=index, linear=linear, child=child)
                peeled = self.geometry.peeled(index, shape)
                if peeled is not None:
                    linear, core, fringes = peeled
                    row = self.best(core)
                    cost = linear + p['rank'] * row['cost']
                    memory = p['rank'] * row['leaf_bytes']
                    for fringe in fringes:
                        row = self.best(fringe)
                        cost += row['cost']
                        memory = max(memory, row['leaf_bytes'])
                    if (cost, memory) < (best['cost'], best['leaf_bytes']):
                        best = dict(cost=cost, leaf_bytes=memory, kind='recursive_peel',
                                    program=index, linear=linear, core=core,
                                    fringes=fringes)
        self.memo[shape] = best
        return best

    def selected_graph(self, roots):
        result = {}
        def visit(shape):
            key = ','.join(map(str, shape))
            if key in result:
                return
            row = self.memo[shape]
            result[key] = row
            if row['kind'] == 'padded_support':
                for child, count in row['children']:
                    visit(child)
            elif row['kind'] == 'recursive_peel':
                visit(row['core'])
                for child in row['fringes']:
                    visit(child)
            elif row['kind'] == 'uniform_padding':
                visit(row['child'])
        for shape in sorted(set(roots)):
            visit(tuple(shape))
        return result
