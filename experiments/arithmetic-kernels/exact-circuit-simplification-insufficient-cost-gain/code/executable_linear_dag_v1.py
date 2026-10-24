"""Exact local linear-DAG compiler with lazy gauges and transposition."""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import numpy as np


@dataclass(frozen=True)
class Wire:
    ref: int
    scale: Fraction = Fraction(1)


ZERO = Wire(-1, Fraction(0))


class Builder:
    def __init__(self, inputs):
        self.inputs = inputs
        self.nodes = [dict(op='input', args=(), literal=i) for i in range(inputs)]
        self.memo = {}

    def emit(self, op, args, literal=None):
        key = (op, tuple(args), literal)
        if key not in self.memo:
            self.memo[key] = len(self.nodes)
            self.nodes.append(dict(op=op, args=tuple(args), literal=literal))
        return self.memo[key]

    def times(self, wire, scalar):
        return Wire(wire.ref, wire.scale*Fraction(scalar)) if scalar else ZERO

    def positive_scale(self, ref, q):
        assert q > 0
        return ref if q == 1 else self.emit('scale', (ref,), q)

    def add(self, a, b):
        if not a.scale:
            return b
        if not b.scale:
            return a
        if a.ref == b.ref:
            return self.times(Wire(a.ref), a.scale+b.scale)
        common = min(abs(a.scale), abs(b.scale))
        ia = self.positive_scale(a.ref, abs(a.scale)/common)
        ib = self.positive_scale(b.ref, abs(b.scale)/common)
        if (a.scale > 0) == (b.scale > 0):
            ia, ib = sorted((ia, ib))
            return Wire(self.emit('add', (ia, ib)), common if a.scale > 0 else -common)
        if a.scale < 0:
            ia, ib = ib, ia
        if ia > ib:
            ia, ib = ib, ia
            common = -common
        return Wire(self.emit('sub', (ia, ib)), common)

    def materialize(self, wire):
        if not wire.scale:
            return Wire(self.emit('zero', ()))
        if wire.scale == 1:
            return Wire(wire.ref)
        op = 'neg' if wire.scale == -1 else 'scale'
        return Wire(self.emit(op, (wire.ref,), wire.scale if op == 'scale' else None))

    def circuit(self, outputs):
        return dict(inputs=self.inputs, nodes=self.nodes, outputs=tuple(outputs))


def clone_into(circuit, builder, inputs):
    assert len(inputs) == circuit['inputs']
    values = {}
    for i, node in enumerate(circuit['nodes']):
        op, args = node['op'], node['args']
        if op == 'input':
            v = inputs[node['literal']]
        elif op == 'add':
            v = builder.add(values[args[0]], values[args[1]])
        elif op == 'sub':
            v = builder.add(values[args[0]], builder.times(values[args[1]], -1))
        elif op in ('scale', 'neg'):
            v = builder.times(values[args[0]], -1 if op == 'neg' else node['literal'])
        elif op == 'zero':
            v = ZERO
        else:
            raise ValueError(op)
        values[i] = v
    return [builder.times(values[w.ref], w.scale) if w.scale else ZERO for w in circuit['outputs']]


def transpose(circuit):
    b = Builder(len(circuit['outputs']))
    adj = {}

    def accumulate(ref, wire):
        if wire.scale:
            adj[ref] = b.add(adj.get(ref, ZERO), wire)

    for i, out in enumerate(circuit['outputs']):
        if out.scale:
            accumulate(out.ref, b.times(Wire(i), out.scale))
    for i in reversed(range(len(circuit['nodes']))):
        dy = adj.get(i, ZERO)
        if not dy.scale:
            continue
        node = circuit['nodes'][i]
        op, args = node['op'], node['args']
        if op in ('add', 'sub'):
            accumulate(args[0], dy)
            accumulate(args[1], b.times(dy, 1 if op == 'add' else -1))
        elif op in ('scale', 'neg'):
            accumulate(args[0], b.times(dy, -1 if op == 'neg' else node['literal']))
        else:
            assert op in ('input', 'zero')
    return b.circuit([adj.get(i, ZERO) for i in range(circuit['inputs'])])


def column_transpose(circuit, rows, columns):
    assert rows*columns == circuit['inputs']
    order = np.arange(rows*columns).reshape(rows, columns).T.ravel()
    b = Builder(circuit['inputs'])
    inputs = [None]*len(order)
    for new, old in enumerate(order):
        inputs[int(old)] = Wire(new)
    return b.circuit(clone_into(circuit, b, inputs))


def row_transpose(circuit, rows, columns):
    assert rows*columns == len(circuit['outputs'])
    order = np.arange(rows*columns).reshape(rows, columns).T.ravel()
    return dict(circuit, outputs=tuple(circuit['outputs'][i] for i in order))


def matrix(circuit):
    n = circuit['inputs']
    values = []
    for node in circuit['nodes']:
        op, args = node['op'], node['args']
        if op == 'input':
            v = np.zeros(n, dtype=object)
            v[node['literal']] = Fraction(1)
        elif op == 'add':
            v = values[args[0]]+values[args[1]]
        elif op == 'sub':
            v = values[args[0]]-values[args[1]]
        elif op in ('scale', 'neg'):
            v = values[args[0]]*(-1 if op == 'neg' else node['literal'])
        else:
            assert op == 'zero'
            v = np.zeros(n, dtype=object)
        values.append(v)
    return np.stack([w.scale*values[w.ref] if w.scale else np.zeros(n, dtype=object)
                     for w in circuit['outputs']])


def physical_bilinear(left, right, output):
    # Unscaled packed inputs; move their exact scalar gauges to the output.
    rank = len(left['outputs'])
    assert len(right['outputs']) == rank == output['inputs']
    b = Builder(rank)
    seeds = [b.times(Wire(i), a.scale*c.scale)
             for i, (a, c) in enumerate(zip(left['outputs'], right['outputs']))]
    outs = clone_into(output, b, seeds)
    physical_output = b.circuit([b.materialize(w) for w in outs])
    physical_left = dict(left, outputs=tuple(Wire(w.ref) for w in left['outputs']))
    physical_right = dict(right, outputs=tuple(Wire(w.ref) for w in right['outputs']))
    assert all(w.scale for w in left['outputs']+right['outputs'])
    return physical_left, physical_right, physical_output


def accounting(circuit):
    live = set()

    def visit(ref):
        if ref in live:
            return
        live.add(ref)
        for arg in circuit['nodes'][ref]['args']:
            visit(arg)

    for w in circuit['outputs']:
        assert w.scale == 1
        visit(w.ref)
    counts = Counter(circuit['nodes'][i]['op'] for i in live)
    copies = 0
    for ref, uses in Counter(w.ref for w in circuit['outputs']).items():
        # One computed node may write straight into its final packed slot;
        # a direct input needs a copy, as do repeated output destinations.
        copies += uses if circuit['nodes'][ref]['op'] in ('input', 'zero') else uses-1
    arithmetic = sum(v for k, v in counts.items() if k not in ('input', 'zero'))
    return dict(arithmetic=arithmetic, packed_copies=copies,
                billed_linear_operations=arithmetic+copies,
                operations=dict(counts), live_nodes=len(live),
                unpriced_temporary_nodes=len(circuit['nodes'])-len(live))


def serialize(circuit):
    return dict(inputs=circuit['inputs'],
                nodes=[dict(n, literal=str(n['literal']) if isinstance(n['literal'], Fraction) else n['literal'])
                       for n in circuit['nodes']],
                outputs=[dict(ref=w.ref, scale=str(w.scale)) for w in circuit['outputs']])
