"""Experimental FNP-only product; exact DAGs, stream8, bounded scratch."""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import prod
import flopscope.numpy as fnp
from streamed_leaf_winograd_v1 import streamed_leaf


@dataclass
class Block:
    data: object
    shape: tuple
    start: object = None
    size: int = 0


class Arena:
    def __init__(self):
        self.capacity = 2**28  # exactly1GiB, padding tracked separately
        self.array = fnp.empty(self.capacity, dtype=fnp.float32)
        self.free = [(0, self.capacity)]
        self.owned = {}
        self.views = {}
        self.padding = {}
        self.padding_bytes = 0
        self.live = self.peak_live = 0

    def allocate(self, shape):
        shape = tuple(shape)
        size = prod(shape)
        for i, (start, length) in enumerate(self.free):
            if length >= size:
                if length == size:
                    self.free.pop(i)
                else:
                    self.free[i] = start+size, length-size
                key = start, shape
                if key not in self.views:
                    self.views[key] = self.array[start:start+size].reshape(shape)
                assert start not in self.owned
                self.owned[start] = size
                self.live += size
                self.peak_live = max(self.peak_live, self.live)
                return Block(self.views[key], shape, start, size)
        raise MemoryError(('arena_fragment_or_capacity', shape, self.live, self.peak_live))

    def release(self, block):
        assert block.start is not None
        assert self.owned.pop(block.start) == block.size
        self.live -= block.size
        self.free.append((block.start, block.size))
        self.free.sort()
        merged = []
        for start, length in self.free:
            if merged and merged[-1][0]+merged[-1][1] == start:
                old_start, old_length = merged[-1]
                merged[-1] = old_start, old_length+length
            else:
                merged.append((start, length))
        self.free = merged

    def padded(self, source, full_shape, key):
        if tuple(source.shape) == tuple(full_shape):
            return source
        key = key, tuple(full_shape), tuple(source.shape)
        if key not in self.padding:
            self.padding[key] = fnp.zeros(full_shape, dtype=fnp.float32)
            self.padding_bytes += prod(full_shape)*4
            assert self.capacity*4+self.padding_bytes <= 1168900096
        target = self.padding[key]
        # Invalid regions are immutable zeros in a dedicated scratch buffer.
        fnp.copyto(target[:, :source.shape[-2], :source.shape[-1]], source)
        return target


def prepare(raw):
    nodes = raw['nodes']
    outputs = [o['ref'] for o in raw['outputs']]
    assert all(Fraction(o['scale']) == 1 for o in raw['outputs'])
    live = set()

    def visit(i):
        if i not in live:
            live.add(i)
            for a in nodes[i]['args']:
                visit(a)

    for i in outputs:
        visit(i)
    uses = Counter(a for i in live for a in nodes[i]['args'])
    destinations = {}
    for i, ref in enumerate(outputs):
        destinations.setdefault(ref, []).append(i)
    operations = []
    for i in sorted(live):
        n = dict(nodes[i])
        if n['op'] == 'scale':
            n['literal'] = float(Fraction(n['literal']))
        operations.append((i, n))
    return dict(operations=operations, uses=uses, destinations=destinations)


def linear(circuit, inputs, outputs, full_shape, arena, accumulate=False):
    """Every internal value freed after its last exact DAG use."""
    remaining = circuit['uses'].copy()
    values, owners = {}, {}
    for ref, node in circuit['operations']:
        op, args = node['op'], node['args']
        destinations = circuit['destinations'].get(ref, ())
        direct = False
        if op == 'input':
            value = inputs[node['literal']]
        else:
            assert op != 'zero'  # not used by the certified fixed primary
            if destinations and not accumulate:
                dest = outputs[destinations[0]]
                direct = tuple(dest.shape) == tuple(full_shape) or remaining[ref] == 0
            if direct:
                value = dest
            else:
                owner = arena.allocate(full_shape)
                owners[ref] = owner
                value = owner.data
            shape = tuple(value.shape)
            operands = [values[a][:, :shape[-2], :shape[-1]] for a in args]
            if op == 'add':
                fnp.add(operands[0], operands[1], out=value)
            elif op == 'sub':
                fnp.subtract(operands[0], operands[1], out=value)
            elif op in ('scale', 'neg'):
                fnp.multiply(operands[0], -1 if op == 'neg' else node['literal'], out=value)
            else:
                raise ValueError(op)
        values[ref] = value
        for j, slot in enumerate(destinations):
            dest = outputs[slot]
            piece = value[:, :dest.shape[-2], :dest.shape[-1]]
            if accumulate:
                fnp.add(dest, piece, out=dest)
            elif not (direct and j == 0):
                fnp.copyto(dest, piece)
        for a in args:
            remaining[a] -= 1
            if remaining[a] == 0:
                if a in owners:
                    arena.release(owners.pop(a))
                values.pop(a)
        if remaining[ref] == 0:
            if ref in owners:
                arena.release(owners.pop(ref))
            values.pop(ref)
    assert not owners and not values


class Product:
    def __init__(self, primary_record):
        assert tuple(primary_record['shape']) == (3, 3, 6)
        assert primary_record['rank'] == 40 and primary_record['passed']
        self.circuits = {k: prepare(primary_record['circuits'][k]) for k in ('L', 'R', 'P')}
        self.arena = Arena()

    def input_blocks(self, a, rows, cols, side):
        batch, m, n = a.shape
        h, w = (m+rows-1)//rows, (n+cols-1)//cols
        values = []
        for i in range(rows):
            for j in range(cols):
                v = a.data[:, i*h:min((i+1)*h, m), j*w:min((j+1)*w, n)]
                values.append(self.arena.padded(v, (batch, h, w), (side, a.shape, i, j)))
        return values, h, w

    def transforms(self, a, b):
        batch, m, k = a.shape
        assert b.shape[0] == batch and b.shape[1] == k
        n = b.shape[2]
        av, aa, kk = self.input_blocks(a, 3, 3, 'A')
        bv, bk, nn = self.input_blocks(b, 3, 6, 'B')
        assert kk == bk
        left = self.arena.allocate((40*batch, aa, kk))
        right = self.arena.allocate((40*batch, kk, nn))
        la = left.data.reshape(40, batch, aa, kk)
        rb = right.data.reshape(40, batch, kk, nn)
        linear(self.circuits['L'], av, [la[i] for i in range(40)], (batch, aa, kk), self.arena)
        linear(self.circuits['R'], bv, [rb[i] for i in range(40)], (batch, kk, nn), self.arena)
        return left, right, (aa, kk, nn)

    def combine(self, products, out, child_dims):
        batch, m, n = out.shape
        aa, _, nn = child_dims
        inputs = products.data.reshape(40, batch, aa, nn)
        outputs = [out.data[:, i*aa:min((i+1)*aa, m), j*nn:min((j+1)*nn, n)]
                   for i in range(3) for j in range(6)]
        linear(self.circuits['P'], [inputs[i] for i in range(40)], outputs,
               (batch, aa, nn), self.arena)

    def leaf(self, a, b, out, consume):
        batch, m, k = a.shape
        n = b.shape[2]
        assert m % 2 == n % 2 == 0
        k0 = k-k % 2
        fringe = None
        if k0 != k:
            fringe = self.arena.allocate(out.shape)
            fnp.matmul(a.data[:, :, k0:], b.data[:, k0:, :], out=fringe.data)
        wa = self.arena.allocate((3, batch, m//2, k0//2))
        wb = self.arena.allocate((3, batch, k0//2, n//2))
        wp = self.arena.allocate((3, batch, m//2, n//2))
        streamed_leaf(a.data[:, :, :k0], b.data[:, :k0, :], out.data, wa.data, wb.data, wp.data)
        self.arena.release(wa)
        self.arena.release(wb)
        self.arena.release(wp)
        if fringe is not None:
            fnp.add(out.data, fringe.data, out=out.data)
            self.arena.release(fringe)
        if consume:
            self.arena.release(a)
            self.arena.release(b)

    def inner(self, a, b, out, remaining, consume):
        if remaining == 0:
            return self.leaf(a, b, out, consume)
        left, right, dims = self.transforms(a, b)
        if consume:
            self.arena.release(a)
            self.arena.release(b)
        products = self.arena.allocate((left.shape[0], dims[0], dims[2]))
        self.inner(left, right, products, remaining-1, True)
        self.combine(products, out, dims)
        self.arena.release(products)

    def __call__(self, a, b, out):
        m, k = tuple(a.shape)
        n = b.shape[1]
        assert tuple(b.shape) == (k, n) and tuple(out.shape) == (m, n)
        assert not self.arena.owned and self.arena.live == 0
        aa = Block(a.reshape(1, m, k), (1, m, k))
        bb = Block(b.reshape(1, k, n), (1, k, n))
        oo = Block(out.reshape(1, m, n), (1, m, n))
        left, right, dims = self.transforms(aa, bb)
        products = self.arena.allocate((40, dims[0], dims[2]))
        for start in range(0, 40, 8):
            av = Block(left.data[start:start+8], (8, dims[0], dims[1]))
            bv = Block(right.data[start:start+8], (8, dims[1], dims[2]))
            cv = Block(products.data[start:start+8], (8, dims[0], dims[2]))
            self.inner(av, bv, cv, 2, False)
        self.combine(products, oo, dims)
        self.arena.release(left)
        self.arena.release(right)
        self.arena.release(products)
        assert not self.arena.owned and self.arena.live == 0
        return out
