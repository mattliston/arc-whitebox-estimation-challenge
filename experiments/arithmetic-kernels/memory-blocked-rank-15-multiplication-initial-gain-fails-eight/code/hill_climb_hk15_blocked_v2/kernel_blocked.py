"""HK15 with analytically bounded leading-product batches in the original arena.

All 15 outer maps and product slots remain live and ordered. Only the three
binary-tree slabs are reused by contiguous batches, without accumulation.
Each matrix keeps the qualified arithmetic and contraction axes unchanged.
Fresh input padding counts toward the fixed complete workspace ceiling.
"""
from collections import Counter
from fractions import Fraction
from math import prod
import flopscope.numpy as fnp

ORIGINAL_ARENA_ELEMENTS = 391602176
BATCH_DIVISORS = (15, 5, 3, 1)


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
        self._borrowed_arena, self._borrowed_start = arena, int(start)
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



def layout_for_batch(record, depth, shape, P):
    """Pure layout formula for a divisor P; this alone does not admit a capacity."""
    assert type(P) is int and P in BATCH_DIVISORS
    assert record['rank'] == 15 and record['passed'] and 1 <= depth <= 6
    shape = tuple(int(x) for x in shape)
    assert len(shape) == 3 and min(shape) > 0
    plans = {s: circuit(record['circuits'][s]) for s in ('L','R','P')}
    assert [plans[s]['operations'] for s in ('L','R','P')] == record['packed_linear_operations']
    r,s,t = record['shape']; m,k,n = shape; u = 1 << depth
    a,b,c = ((x+q*u-1)//(q*u)*u for x,q in zip(shape,(r,s,t)))
    M,K,N = r*a,s*b,t*c
    scratch = max(plans[z]['scratch']*x*y for z,x,y in (('L',a,b),('R',b,c),('P',a,c)))
    output_pad = M*N if (M,N)!=(m,n) else 0
    fresh = (M*K if (M,K)!=(m,k) else 0) + (K*N if (K,N)!=(k,n) else 0)
    per_product_tree = 7**depth*((a//u)*(b//u)+(b//u)*(c//u)+(a//u)*(c//u))
    fixed_workspace = 15*(a*b+b*c+a*c) + scratch + output_pad
    batch_size = P
    batch_count = 15//batch_size
    segments = {}; cursor = 0
    def take(name, dims):
        nonlocal cursor
        count = prod(dims)
        segments[name] = dict(start=cursor, elements=count, shape=tuple(dims))
        cursor += count
    for j,(x,y) in enumerate(((a,b),(b,c),(a,c))):
        take('tree'+str(j), (batch_size,)+(7,)*depth+(x//u,y//u))
    # Every packed product output is beyond the WHOLE reusable tree interval.
    # A later tree batch cannot overwrite an earlier completed product slot.
    take('left',(15,a,b)); take('right',(15,b,c)); take('products',(15,a,c))
    take('scratch',(scratch,))
    if output_pad: take('pad_c',(M,N))
    assert cursor == batch_size*per_product_tree+fixed_workspace
    side_bind = sum(segments['tree'+str(j)]['elements'] for j in range(3))
    side_bind += batch_size*7*4**(depth-1)*((a//u)*(b//u)+(b//u)*(c//u)+(a//u)*(c//u))
    descriptor_bind = 15*(a*b+b*c+a*c)
    descriptor_bind += sum(plans[z]['scratch']*x*y for z,x,y in (('L',a,b),('R',b,c),('P',a,c)))
    descriptor_bind += output_pad
    return dict(shape=shape, child=(a,b,c), padded=(M,K,N), plans=plans,
                segments=segments, required_arena_elements=cursor,
                fresh_padding_elements=fresh, complete_workspace_elements=cursor+fresh,
                per_product_tree_elements=per_product_tree,
                fixed_workspace_elements=fixed_workspace,
                batch_size=batch_size, batch_count=batch_count,
                batch_view_count=0 if batch_size==15 else 3*batch_count,
                batch_view_bind_flops=0, side_bind_flops=side_bind,
                descriptor_bind_flops=descriptor_bind)


def borrowed_layout(record, depth, shape, capacity_elements=ORIGINAL_ARENA_ELEMENTS):
    """Largest batch divisor fitting tree + full maps/work + fresh input pads.

    Production passes the original fixed arena capacity. A smaller actual
    arena follows this same metadata-only rule, including in integrity fixtures.
    """
    assert type(capacity_elements) is int and 0 < capacity_elements <= ORIGINAL_ARENA_ELEMENTS
    full = layout_for_batch(record,depth,shape,15)
    eligible = [P for P in BATCH_DIVISORS
                if P*full['per_product_tree_elements']+full['fixed_workspace_elements']
                   +full['fresh_padding_elements'] <= capacity_elements]
    assert eligible, 'complete HK workspace exceeds capacity even at P=1'
    layout = full if eligible[0]==15 else layout_for_batch(record,depth,shape,eligible[0])
    layout['original_arena_elements'] = ORIGINAL_ARENA_ELEMENTS
    layout['capacity_elements'] = capacity_elements
    assert layout['complete_workspace_elements'] <= capacity_elements
    return layout



class BorrowedProduct:
    """Bind once per fixed descriptor/generation; caller caches this whole object.

    side_provider(arena,start,kind,P,L,rows,cols) returns an exact Side. The
    provider may share only the same arena identity and offset/layout key.
    bind_flops is the nominal independent cold price; pooled setup instead
    counts unique side_bind work plus descriptor_bind_flops for every product.
    """
    def __init__(self, record, depth, shape, arena, side_provider=None):
        assert len(arena.shape)==1 and str(arena.dtype)=='float32'
        self.layout = layout = borrowed_layout(record,depth,shape,
            capacity_elements=min(ORIGINAL_ARENA_ELEMENTS,int(arena.shape[0])))
        self.record,self.depth,self.shape = record,depth,layout['shape']
        self.child,self.padded,self.plans = layout['child'],layout['padded'],layout['plans']
        self.arena = arena
        self.batch_size,self.batch_count = layout['batch_size'],layout['batch_count']
        self.batch_view_count = layout['batch_view_count']
        self.batch_view_bind_flops = layout['batch_view_bind_flops']
        self.required_arena_elements = self.workspace_elements = layout['required_arena_elements']
        assert int(arena.shape[0]) >= self.required_arena_elements
        self.borrowed_bytes = 4*self.required_arena_elements
        self.fresh_padding_bytes = 4*layout['fresh_padding_elements']
        self.workspace_bytes = self.borrowed_bytes+self.fresh_padding_bytes
        self.side_bind_flops = layout['side_bind_flops']
        self.descriptor_bind_flops = layout['descriptor_bind_flops']
        self.bind_flops = self.side_bind_flops+self.descriptor_bind_flops
        segments = layout['segments']
        def view(name):
            d=segments[name]
            return fnp.reshape(arena[d['start']:d['start']+d['elements']],d['shape'])
        provider = Side if side_provider is None else side_provider
        a,b,c = self.child
        self.sides=[]
        for j,(x,y) in enumerate(((a,b),(b,c),(a,c))):
            start=segments['tree'+str(j)]['start']
            side=provider(arena,start,j,self.batch_size,depth,x,y)
            assert isinstance(side,Side)
            assert side._borrowed_arena is arena and side._borrowed_start==start
            assert (side.kind,side.P,side.L,side.rows,side.cols)==(j,self.batch_size,depth,x,y)
            self.sides.append(side)
        self.left,self.right,self.products = view('left'),view('right'),view('products')
        d=segments['scratch'];self.scratch_size=d['elements']
        self.scratch=arena[d['start']:d['start']+d['elements']]
        self.work={z:[fnp.reshape(self.scratch[j*x*y:(j+1)*x*y],(x,y))
                      for j in range(self.plans[z]['scratch'])]
                   for z,x,y in (('L',a,b),('R',b,c),('P',a,c))}
        self.pad_c=view('pad_c') if 'pad_c' in segments else None
        self.left_slots=[self.left[j] for j in range(15)]
        self.right_slots=[self.right[j] for j in range(15)]
        self.product_slots=[self.products[j] for j in range(15)]
        if self.batch_size==15:
            self.batches=[(self.left,self.right,self.products)]
        else:
            P=self.batch_size
            self.batches=[(self.left[j:j+P],self.right[j:j+P],self.products[j:j+P])
                          for j in range(0,15,P)]
        assert len(self.batches)==self.batch_count

    def __call__(self, aa, bb, out):
        m,k,n=self.shape; a,b,c=self.child; r,s,t=self.record['shape']; M,K,N=self.padded
        assert tuple(aa.shape)==(m,k) and tuple(bb.shape)==(k,n) and tuple(out.shape)==(m,n)
        assert str(aa.dtype)==str(bb.dtype)==str(out.dtype)=='float32'
        hold=[]
        try:
            # Shared-arena padding halos can be stale after any other owner.
            # These allocations are deliberately fresh, zero-filled and held.
            if (M,K)!=(m,k):
                padded=fnp.zeros((M,K),dtype=fnp.float32); piece=padded[:m,:k]
                hold.extend((padded,piece)); hold.append(fnp.copyto(piece,aa)); aa=padded
            if (K,N)!=(k,n):
                padded=fnp.zeros((K,N),dtype=fnp.float32); piece=padded[:k,:n]
                hold.extend((padded,piece)); hold.append(fnp.copyto(piece,bb)); bb=padded
            target=self.pad_c if self.pad_c is not None else out
            av=[aa[i*a:(i+1)*a,j*b:(j+1)*b] for i in range(r) for j in range(s)]
            bv=[bb[i*b:(i+1)*b,j*c:(j+1)*c] for i in range(s) for j in range(t)]
            cv=[target[i*a:(i+1)*a,j*c:(j+1)*c] for i in range(r) for j in range(t)]
            hold.extend((av,bv,cv))
            linear(self.plans['L'],av,self.left_slots,self.work['L'],hold)
            linear(self.plans['R'],bv,self.right_slots,self.work['R'],hold)
            for left,right,products in self.batches:
                self.sides[0].forward(left,hold); self.sides[1].forward(right,hold)
                hold.append(fnp.matmul(self.sides[0].T,self.sides[1].T,out=self.sides[2].T))
                self.sides[2].unwind(products,hold)
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
