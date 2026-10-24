"""Symbolic accounting and complete live-storage audit; no large arrays."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from math import prod
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent
OUT = HERE/'smirnov_global_batch_static_v1.json'
assert not OUT.exists()


class Symbol:
    def __init__(self, shape, meter):
        self.shape = tuple(map(int, shape))
        self.meter = meter

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),)*(len(self.shape)-len(key)+1) + key[i+1:]
        key = key + (slice(None),)*(len(self.shape)-len(key))
        shape = []
        for size, piece in zip(self.shape, key):
            if isinstance(piece, int):
                assert -size <= piece < size
            else:
                start, stop, step = piece.indices(size)
                shape.append(len(range(start, stop, step)))
        assert len(key) == len(self.shape)
        self.meter.views += 1
        return Symbol(shape, self.meter)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        assert prod(shape) == prod(self.shape), (self.shape, shape)
        self.meter.charge('reshape', prod(shape))
        return Symbol(shape, self.meter)


class Meter:
    float32 = 'float32'

    def __init__(self):
        self.reset()

    def reset(self):
        self.cost = Counter()
        self.calls = Counter()
        self.views = 0

    def charge(self, name, count):
        self.cost[name] += int(count)
        self.calls[name] += 1

    def empty(self, shape, dtype=None):
        return Symbol((shape,) if isinstance(shape, int) else shape, self)

    def zeros(self, shape, dtype=None):
        self.charge('zeros', prod(shape))
        return Symbol(shape, self)

    def copyto(self, destination, source):
        assert destination.shape == source.shape
        self.charge('copyto', prod(destination.shape))

    def point(self, name, a, b, out):
        if isinstance(a, Symbol):
            assert a.shape == out.shape
        if isinstance(b, Symbol):
            assert b.shape == out.shape
        self.charge(name, prod(out.shape))

    def add(self, a, b, out):
        self.point('add', a, b, out)

    def subtract(self, a, b, out):
        self.point('subtract', a, b, out)

    def multiply(self, a, b, out):
        self.point('multiply', a, b, out)

    def matmul(self, a, b, out):
        assert a.shape[:-2] == b.shape[:-2] == out.shape[:-2]
        batch = prod(a.shape[:-2])
        m, k = a.shape[-2:]
        assert b.shape[-2] == k and out.shape[-2:] == (m, b.shape[-1])
        n = b.shape[-1]
        self.charge('matmul', batch*(2*m*k*n-m*n))

    def record(self):
        return dict(flops=sum(self.cost.values()), operation_flops=dict(self.cost),
                    operation_calls=dict(self.calls), counted_calls=sum(self.calls.values()),
                    uncounted_views=self.views)


meter = Meter()
environment = dict(fnp=meter, __name__=__name__)
leaf_source = (HERE/'streamed_leaf_winograd_v1.py').read_text()
exec(compile(leaf_source.replace('import flopscope.numpy as fnp', ''),
             str(HERE/'streamed_leaf_winograd_v1.py'), 'exec'), environment)
kernel_source = (HERE/'smirnov_stream8_native_v1.py').read_text()
assert hashlib.sha256(kernel_source.encode()).hexdigest() == '7c9ca40924410a7d5b6280c40bf500f78d9c935c831914e90047c549de430c74'
kernel_source = kernel_source.replace('import flopscope.numpy as fnp', '')
kernel_source = kernel_source.replace('from streamed_leaf_winograd_v1 import streamed_leaf', '')
exec(compile(kernel_source, str(HERE/'smirnov_stream8_native_v1.py'), 'exec'), environment)
Block, Arena, Product = [environment[k] for k in ('Block', 'Arena', 'Product')]
linear, streamed_leaf = [environment[k] for k in ('linear', 'streamed_leaf')]


class TraceArena(Arena):
    def __init__(self):
        # Symbolic address space only; the final concrete high-water mark is
        # the proposed retained arena, NOT this virtual capacity.
        super().__init__()
        self.capacity = 2**34
        self.array = meter.empty(self.capacity)
        self.free = [(0, self.capacity)]
        self.serial = 0
        self.clock = 0
        self.intervals = []
        self.open_intervals = {}
        self.high_water = 0
        self.peak_records = []

    def allocate(self, shape):
        old_peak = self.peak_live
        block = super().allocate(shape)
        self.clock += 1
        self.serial += 1
        record = dict(id=self.serial, shape=tuple(shape), size=block.size,
                      start_time=self.clock, end_time=None,
                      first_fit_offset=block.start)
        self.intervals.append(record)
        self.open_intervals[block.start] = record
        self.high_water = max(self.high_water, block.start+block.size)
        if self.peak_live > old_peak:
            self.peak_records.append(dict(time=self.clock, live_elements=self.live,
                                         shapes=[r['shape'] for r in self.open_intervals.values()]))
        return block

    def release(self, block):
        self.clock += 1
        self.open_intervals.pop(block.start)['end_time'] = self.clock
        super().release(block)

    def padded(self, source, full_shape, key):
        # Preserve exact dedicated-zero-buffer behavior, but do not impose
        # the old arbitrary1GiB arena on the symbolic diagnostic address map.
        if tuple(source.shape) == tuple(full_shape):
            return source
        key = key, tuple(full_shape), tuple(source.shape)
        if key not in self.padding:
            self.padding[key] = meter.zeros(full_shape)
            self.padding_bytes += prod(full_shape)*4
        target = self.padding[key]
        meter.copyto(target[:, :source.shape[-2], :source.shape[-1]], source)
        return target

    def reset_trace(self):
        assert not self.owned
        self.serial = self.clock = self.high_water = self.peak_live = 0
        self.intervals = []
        self.open_intervals = {}
        self.peak_records = []


class GlobalBatch(Product):
    def __init__(self, record):
        super().__init__(record)
        self.arena = TraceArena()

    def transform_one(self, a, rows, cols, side, circuit, consume):
        batch = a.shape[0]
        values, height, width = self.input_blocks(a, rows, cols, side)
        transformed = self.arena.allocate((40*batch, height, width))
        pieces = transformed.data.reshape(40, batch, height, width)
        linear(self.circuits[circuit], values, [pieces[i] for i in range(40)],
               (batch, height, width), self.arena)
        if consume:
            self.arena.release(a)
        return transformed

    def delayed(self, a, b, remaining, consume):
        batch, m, k = a.shape
        n = b.shape[-1]
        if not remaining:
            out = self.arena.allocate((batch,m,n))
            for begin in range(0,batch,12800):
                end = min(batch,begin+12800)
                ac = Block(a.data[begin:end], (end-begin,m,k))
                bc = Block(b.data[begin:end], (end-begin,k,n))
                oc = Block(out.data[begin:end], (end-begin,m,n))
                self.leaf_late_fringe(ac,bc,oc)
            if consume:
                self.arena.release(a)
                self.arena.release(b)
            return out
        left = self.transform_one(a,3,3,'A','L',consume)
        right = self.transform_one(b,3,6,'B','R',consume)
        dims = left.shape[-2],left.shape[-1],right.shape[-1]
        products = self.delayed(left,right,remaining-1,True)
        out = self.arena.allocate((batch,m,n))
        self.combine(products,out,dims)
        self.arena.release(products)
        return out

    def leaf_late_fringe(self, a,b,out):
        batch,m,k = a.shape
        n = b.shape[-1]
        k0 = k-k%2
        assert m%2 == n%2 == 0
        wa = self.arena.allocate((3,batch,m//2,k0//2))
        wb = self.arena.allocate((3,batch,k0//2,n//2))
        wp = self.arena.allocate((3,batch,m//2,n//2))
        streamed_leaf(a.data[:,:,:k0],b.data[:,:k0,:],out.data,wa.data,wb.data,wp.data)
        for work in (wa,wb,wp):
            self.arena.release(work)
        if k0 != k:
            fringe = self.arena.allocate(out.shape)
            meter.matmul(a.data[:,:,k0:],b.data[:,k0:,:],out=fringe.data)
            meter.add(out.data,fringe.data,out=out.data)
            self.arena.release(fringe)

    def __call__(self,a,b,out):
        m,k = a.shape
        n = b.shape[-1]
        aa = Block(a.reshape(1,m,k),(1,m,k))
        bb = Block(b.reshape(1,k,n),(1,k,n))
        oo = Block(out.reshape(1,m,n),(1,m,n))
        left = self.transform_one(aa,3,3,'A','L',False)
        right = self.transform_one(bb,3,6,'B','R',False)
        dims = left.shape[-2],left.shape[-1],right.shape[-1]
        products = self.delayed(left,right,2,True)
        self.combine(products,oo,dims)
        self.arena.release(products)
        assert not self.arena.owned and self.arena.live == 0
        return out


def place_largest_first(intervals):
    ordered = sorted(intervals,key=lambda r:(-r['size'],r['start_time'],r['id']))
    placed = []
    for record in ordered:
        overlaps = [r for r in placed if record['start_time'] < r['end_time'] and r['start_time'] < record['end_time']]
        candidate = 0
        for other in sorted(overlaps,key=lambda r:r['offset']):
            if candidate+record['size'] <= other['offset']:
                break
            candidate = max(candidate,other['offset']+other['size'])
        placed.append(dict(record,offset=candidate))
    for i,record in enumerate(placed):
        for other in placed[:i]:
            time_overlap = record['start_time'] < other['end_time'] and other['start_time'] < record['end_time']
            memory_overlap = record['offset'] < other['offset']+other['size'] and other['offset'] < record['offset']+record['size']
            assert not (time_overlap and memory_overlap)
    high_water = max(r['offset']+r['size'] for r in placed)
    return high_water,placed


certificate = json.loads((HERE/'smirnov_executable_transpose_certificate_v1.json').read_text())
assert certificate['passed']
primary = next(r for r in certificate['records'] if r['name']=='signed_executable_0')
started = time.time()
records=[]
for shape,mode in (((216,216,432),'frozen'),((1024,832,8192),'frozen'),((1024,832,8192),'global_batch')):
    kernel = GlobalBatch(primary) if mode=='global_batch' else Product(primary)
    if mode=='frozen':
        kernel.arena = TraceArena()
    m,k,n=shape
    a,b,out=[Symbol(s,meter) for s in ((m,k),(k,n),(m,n))]
    for repetition in ('cold','warm'):
        kernel.arena.reset_trace()
        meter.reset()
        kernel(a,b,out)
        record = dict(mode=mode,shape=shape,repetition=repetition,**meter.record(),
                      peak_live_arena_bytes=4*kernel.arena.peak_live,
                      first_fit_arena_bytes=4*kernel.arena.high_water,
                      dedicated_padding_bytes=kernel.arena.padding_bytes,
                      largest_peak_state=kernel.arena.peak_records[-1],
                      intervals=kernel.arena.intervals)
        if mode=='global_batch':
            high, placement=place_largest_first(kernel.arena.intervals)
            record.update(largest_first_arena_bytes=4*high,
                          largest_first_total_workspace_bytes=4*high+kernel.arena.padding_bytes,
                          logical_live_plus_padding_bytes=4*kernel.arena.peak_live+kernel.arena.padding_bytes,
                          placement_verified=True,placement=placement)
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('intervals','placement','largest_peak_state')}),flush=True)

expected = {(216,216,432):(50036224,45963136), (1024,832,8192):(6834384092,6611249012)}
prerequisites=[]
for row in records[:4]:
    index=0 if row['repetition']=='cold' else 1
    bill=expected[tuple(row['shape'])][index]
    prerequisites.append(dict(shape=row['shape'],repetition=row['repetition'],
                              expected_flops=bill,actual_flops=row['flops'],passed=row['flops']==bill))
full=records[3]
prerequisites.append(dict(name='frozen_memory',passed=full['peak_live_arena_bytes']==595483520 and full['dedicated_padding_bytes']==20499472))
prerequisite_pass=all(r['passed'] for r in prerequisites)
trial=records[-1]
result=dict(prerequisite_pass=prerequisite_pass,prerequisites=prerequisites,
            workspace_limit_bytes=1168900096,
            static_memory_pass=bool(prerequisite_pass and trial['largest_first_total_workspace_bytes']<=1168900096),
            records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            utc_finished=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-started,
            peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            qualification='No large arrays or kernel arithmetic executed. Frozen schedule accounting and memory prerequisites must pass before relying on the new trace. Largest-first is one verified concrete nonoverlapping lifetime placement, not an optimal-placement proof. Exact supported operations mirror existing FNP batched products, elementary views and copies; numerical/runtime/full-card validity remains untested.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('records',)},indent=2),flush=True)
