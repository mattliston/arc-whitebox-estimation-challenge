"""Isolated certified HK15 maps and P-leading fused binary tree. FNP only.

Binding is separate, paid work. This module has no estimator dispatch, target
access, resident-transform promises, or unmetered numerical operations.
"""
from collections import Counter
from fractions import Fraction
from math import prod
import flopscope.numpy as fnp


def circuit(raw):
    nodes = raw['nodes']
    assert all(Fraction(o['scale']) == 1 for o in raw['outputs'])
    destinations = {}
    for j, out in enumerate(raw['outputs']):
        destinations.setdefault(out['ref'], []).append(j)
    live = set()
    def visit(i):
        if i not in live:
            live.add(i)
            for a in nodes[i]['args']: visit(a)
    for i in destinations: visit(i)
    remaining = Counter(a for i in live for a in nodes[i]['args'])
    locations, owners, free, steps = {}, {}, [], []
    capacity = 0
    operations = 0
    for i in sorted(live):
        node = nodes[i]; op = node['op']; args = node['args']
        dest = destinations.get(i, [])
        if op == 'input':
            location = ('input', node['literal'])
        else:
            assert op in ('add', 'sub', 'neg')
            if dest:
                location = ('output', dest[0])
            else:
                if free: slot = free.pop(0)
                else: slot = capacity; capacity += 1
                owners[i] = slot; location = ('scratch', slot)
            operations += 1
        copies = dest if op == 'input' else dest[1:]
        operations += len(copies)
        steps.append((op, location, tuple(locations[a] for a in args), tuple(copies)))
        locations[i] = location
        # New destination is assigned BEFORE reclaiming this instruction's inputs.
        for a in args:
            remaining[a] -= 1
            if remaining[a] == 0 and a in owners:
                free.append(owners.pop(a)); free.sort()
        if remaining[i] == 0 and i in owners:
            free.append(owners.pop(i)); free.sort()
    assert not owners
    return dict(steps=steps, scratch=capacity, operations=operations)


def linear(plan, inputs, outputs, scratch, hold):
    banks = dict(input=inputs, output=outputs, scratch=scratch)
    def value(location): return banks[location[0]][location[1]]
    for op, dest, args, copies in plan['steps']:
        target = value(dest)
        if op == 'add': hold.append(fnp.add(value(args[0]), value(args[1]), out=target))
        elif op == 'sub': hold.append(fnp.subtract(value(args[0]), value(args[1]), out=target))
        elif op == 'neg': hold.append(fnp.negative(value(args[0]), out=target))
        else: assert op == 'input'
        for j in copies: hold.append(fnp.copyto(outputs[j], target))


class Side:
    """Exact dense tree storage; P is never mixed with recursive bit axes."""
    def __init__(self, arena, start, kind, P, L, rows, cols):
        self.kind, self.P, self.L = kind, P, L
        self.rows, self.cols = rows, cols
        u = 1 << L; r, c = rows // u, cols // u
        size = P * 7**L * r * c
        self.T = fnp.reshape(arena[start:start + size], (P,) + (7,) * L + (r, c))
        data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
        self.chunks = [None, None]; levels = {}
        for j in range(2, L + 1):
            nr = L - j
            V = self.T[(slice(None),) + (data,) * nr]
            self.chunks.append(fnp.unstack(V, axis=nr + 1))
            levels[j] = fnp.moveaxis(V, nr + 1, 0)
        nr = L - 1
        R = self.T[(slice(None),) + (data,) * nr]
        V = fnp.reshape(R, (P,) + (2, 2) * nr + (7, r, c))
        ch = 1 + 2 * nr
        ri = [1 + 2 * (L - j) for j in range(2, L + 1)]
        ci = [x + 1 for x in ri]
        if kind == 2: ri, ci = ci, ri
        G = fnp.transpose(V, [ch, 0] + ri + [ch + 1] + ci + [ch + 2])
        self.chunks[1] = fnp.unstack(G, axis=0)
        if kind == 0:
            self.pair = fnp.transpose(G[0:3:2], [1] + list(range(2, L + 2)) + [0] + list(range(L + 2, 2 * L + 2)))
        elif kind == 1:
            self.pair = fnp.moveaxis(G[0:3:2], 0, 1)
        else: self.pair = None
        self.basis = []
        for j in range(2, L + 1):
            ax = L - j
            idx = [data] * L
            if kind == 2: idx[-1] = slice(None)
            lo, hi = list(idx), list(idx)
            lo[ax], hi[ax] = (4, 6) if kind == 0 else (1, 3)
            self.basis.append((self.T[(slice(None),) + tuple(lo)], self.T[(slice(None),) + tuple(hi)]))
        self.ops = []
        if kind == 0:
            for j in range(1, L + 1):
                X = self.chunks[j]; V = G if j == 1 else levels[j]
                self.ops += [(fnp.subtract, X[4], X[0], X[5]),
                             (fnp.subtract, V[2:7:4], X[5], V[3:0:-2])]
        elif kind == 1:
            for j in range(1, L + 1):
                Y = self.chunks[j]; V = G if j == 1 else levels[j]
                self.ops += [(fnp.add, Y[1], Y[0], Y[5]),
                             (fnp.subtract, V[5:2:-2], V[2:6:3], V[6:3:-2])]
        else:
            for j in range(L, 1, -1):
                Z = self.chunks[j]; V = levels[j]
                self.ops += [(fnp.add, Z[0], V[5:1:-3], V[5::-5]),
                             (fnp.add, Z[5], Z[4], Z[5]),
                             (fnp.add, Z[5], V[3:0:-2], V[2:4]),
                             (fnp.add, Z[4], Z[6], Z[1])]

    def forward(self, packed, hold):
        P, L, rows, cols = self.P, self.L, self.rows, self.cols
        u = 1 << L
        V = fnp.reshape(packed, (P,) + (2,) * L + (rows // u,) + (2,) * L + (cols // u,))
        X = self.chunks[1]
        if self.kind == 0:
            A0, A1 = fnp.unstack(V, axis=1)
            a21, a22 = fnp.unstack(A1, axis=L + 1)
            hold.extend((V,A0,A1,a21,a22))
            hold.append(fnp.copyto(self.pair, A0)); hold.append(fnp.copyto(X[6], a22)); hold.append(fnp.add(a21, a22, out=X[4]))
            for lo, hi in self.basis: hold.append(fnp.add(lo, hi, out=lo))
        else:
            B0, B1 = fnp.unstack(V, axis=L + 2)
            b12, b22 = fnp.unstack(B1, axis=1)
            hold.extend((V,B0,B1,b12,b22))
            hold.append(fnp.copyto(self.pair, B0)); hold.append(fnp.copyto(X[3], b22)); hold.append(fnp.subtract(b22, b12, out=X[1]))
            for lo, hi in self.basis: hold.append(fnp.subtract(hi, lo, out=lo))
        for f, x, y, out in self.ops: hold.append(f(x, y, out=out))

    def unwind(self, packed, hold):
        for f, x, y, out in self.ops: hold.append(f(x, y, out=out))
        for lo, hi in self.basis: hold.append(fnp.subtract(hi, lo, out=lo))
        P, L, m, n = self.P, self.L, self.rows, self.cols; u = 1 << L
        V = fnp.reshape(packed, (P,) + (2,) * L + (m // u,) + (2,) * L + (n // u,))
        C0, C1 = fnp.unstack(V, axis=1)
        tl, tr = fnp.unstack(C0, axis=L + 1); bl, br = fnp.unstack(C1, axis=L + 1)
        hold.extend((V,C0,C1,tl,tr,bl,br))
        Z = self.chunks[1]
        hold.append(fnp.add(Z[0], Z[5], out=Z[5])); hold.append(fnp.add(Z[0], Z[2], out=tl))
        hold.append(fnp.add(Z[5], Z[4], out=Z[5])); hold.append(fnp.add(Z[5], Z[3], out=tr))
        hold.append(fnp.add(Z[5], Z[1], out=br)); hold.append(fnp.add(Z[4], Z[6], out=Z[4])); hold.append(fnp.subtract(br, Z[4], out=bl))


class Product:
    """One fixed shape and orientation, one compact tree arena and scratch pool."""
    def __init__(self, record, depth, shape):
        assert record['rank'] == 15 and record['passed'] and 1 <= depth <= 6
        self.record, self.depth, self.shape = record, depth, tuple(shape)
        self.plans = {s: circuit(record['circuits'][s]) for s in ('L', 'R', 'P')}
        assert [self.plans[s]['operations'] for s in ('L','R','P')] == record['packed_linear_operations']
        r, s, t = record['shape']; m, k, n = self.shape; u = 1 << depth
        a, b, c = ((x + q * u - 1) // (q * u) * u for x, q in zip(self.shape, (r, s, t)))
        self.child = a, b, c; self.padded = M, K, N = r*a, s*b, t*c
        self.left = fnp.empty((15, a, b), dtype=fnp.float32)
        self.right = fnp.empty((15, b, c), dtype=fnp.float32)
        self.products = fnp.empty((15, a, c), dtype=fnp.float32)
        sizes = [15 * 7**depth * (x//u) * (y//u) for x,y in ((a,b),(b,c),(a,c))]
        self.arena = fnp.empty(sum(sizes), dtype=fnp.float32)
        self.sides = [Side(self.arena, sum(sizes[:j]), j, 15, depth, x, y)
                      for j,(x,y) in enumerate(((a,b),(b,c),(a,c)))]
        self.scratch_size = max(self.plans[z]['scratch'] * x*y for z,x,y in (('L',a,b),('R',b,c),('P',a,c)))
        self.scratch = fnp.empty(self.scratch_size, dtype=fnp.float32)
        self.work = {z: [fnp.reshape(self.scratch[j*x*y:(j+1)*x*y], (x,y))
                        for j in range(self.plans[z]['scratch'])]
                     for z,x,y in (('L',a,b),('R',b,c),('P',a,c))}
        self.pad_a = fnp.zeros((M,K), dtype=fnp.float32) if (M,K)!=(m,k) else None
        self.pad_b = fnp.zeros((K,N), dtype=fnp.float32) if (K,N)!=(k,n) else None
        self.pad_c = fnp.empty((M,N), dtype=fnp.float32) if (M,N)!=(m,n) else None
        self.left_slots = [self.left[j] for j in range(15)]
        self.right_slots = [self.right[j] for j in range(15)]
        self.product_slots = [self.products[j] for j in range(15)]
        self.workspace_bytes = 4*(sum(sizes)+15*(a*b+b*c+a*c)+self.scratch_size
            +(M*K if self.pad_a is not None else 0)+(K*N if self.pad_b is not None else 0)
            +(M*N if self.pad_c is not None else 0))
        self.bind_flops = sum(sizes) + 15*7*4**(depth-1)*((a//u)*(b//u)+(b//u)*(c//u)+(a//u)*(c//u))
        self.bind_flops += sum(self.plans[z]['scratch']*x*y for z,x,y in (('L',a,b),('R',b,c),('P',a,c)))

    def __call__(self, aa, bb, out):
        m,k,n=self.shape; a,b,c=self.child; r,s,t=self.record['shape']
        assert tuple(aa.shape)==(m,k) and tuple(bb.shape)==(k,n) and tuple(out.shape)==(m,n)
        assert str(aa.dtype)==str(bb.dtype)==str(out.dtype)=='float32'
        hold=[]
        try:
            if self.pad_a is not None:
                piece=self.pad_a[:m,:k]; hold.append(piece)
                hold.append(fnp.copyto(piece, aa)); aa=self.pad_a
            if self.pad_b is not None:
                piece=self.pad_b[:k,:n]; hold.append(piece)
                hold.append(fnp.copyto(piece, bb)); bb=self.pad_b
            target=self.pad_c if self.pad_c is not None else out
            av=[aa[i*a:(i+1)*a,j*b:(j+1)*b] for i in range(r) for j in range(s)]
            bv=[bb[i*b:(i+1)*b,j*c:(j+1)*c] for i in range(s) for j in range(t)]
            cv=[target[i*a:(i+1)*a,j*c:(j+1)*c] for i in range(r) for j in range(t)]
            hold.extend((av,bv,cv))
            linear(self.plans['L'],av,self.left_slots,self.work['L'],hold)
            linear(self.plans['R'],bv,self.right_slots,self.work['R'],hold)
            self.sides[0].forward(self.left,hold); self.sides[1].forward(self.right,hold)
            hold.append(fnp.matmul(self.sides[0].T,self.sides[1].T,out=self.sides[2].T))
            self.sides[2].unwind(self.products,hold)
            linear(self.plans['P'],self.product_slots,cv,self.work['P'],hold)
            if self.pad_c is not None:
                piece=self.pad_c[:m,:n]; hold.append(piece); hold.append(fnp.copyto(out,piece))
        finally:hold.clear()
        return out


def warm_flops(record, depth, shape):
    u=1<<depth; r,s,t=record['shape']; m,k,n=shape
    a,b,c=((x+q*u-1)//(q*u)*u for x,q in zip(shape,(r,s,t)))
    g=Fraction(7,4)**depth; load=g+Fraction(depth+3,4)
    unwind=2*g-Fraction(3,4)+Fraction(7*(depth-1),16)
    core=15*(load*(a*b+b*c)+unwind*a*c+7**depth*(a//u)*(c//u)*(2*(b//u)-1))
    la,lb,lc=record['packed_linear_operations']; M,K,N=r*a,s*b,t*c
    pad=(m*k if (M,K)!=(m,k) else 0)+(k*n if (K,N)!=(k,n) else 0)+(m*n if (M,N)!=(m,n) else 0)
    return int(core+la*a*b+lb*b*c+lc*a*c+pad)
