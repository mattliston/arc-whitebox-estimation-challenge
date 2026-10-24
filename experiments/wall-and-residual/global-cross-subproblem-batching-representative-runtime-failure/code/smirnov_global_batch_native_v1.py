"""FNP-only global batch product with a verified static lifetime placement."""
import ast
from pathlib import Path
import flopscope.numpy as fnp
from smirnov_stream8_native_v1 import Arena, Block, Product, linear, prepare
from streamed_leaf_winograd_v1 import streamed_leaf


class StaticArena(Arena):
    def __init__(self, shared_array, placement):
        self.plan = sorted(placement,key=lambda row:row['id'])
        self.capacity = max(row['offset']+row['size'] for row in self.plan)
        assert int(shared_array.shape[0]) >= self.capacity
        self.array = shared_array
        self.owned = {}
        self.views = {}
        self.padding = {}
        self.padding_bytes = 0
        self.live = self.peak_live = 0
        self.cursor = self.clock = 0

    def begin(self):
        assert not self.owned and self.live == 0
        self.cursor = self.clock = 0

    def allocate(self,shape):
        row = self.plan[self.cursor]
        self.cursor += 1
        self.clock += 1
        shape = tuple(shape)
        assert shape == tuple(row['shape']) and self.clock == row['start_time']
        start,size = row['offset'],row['size']
        assert start not in self.owned
        key = start,shape
        if key not in self.views:
            self.views[key] = self.array[start:start+size].reshape(shape)
        self.owned[start] = row
        self.live += size
        self.peak_live = max(self.peak_live,self.live)
        return Block(self.views[key],shape,start,size)

    def release(self,block):
        row = self.owned.pop(block.start)
        self.clock += 1
        assert row['size'] == block.size and row['end_time'] == self.clock
        self.live -= block.size

    def finish(self):
        assert not self.owned and self.live == 0 and self.cursor == len(self.plan)


# Compile exactly the already-audited schedule, changing only the symbolic
# arena to the concrete predeclared placement and its call-boundary checks.
_here = Path(__file__).resolve().parent
_source = (_here/'audit_smirnov_global_batch_static_v1.py').read_text()
_lines = _source.splitlines(keepends=True)
_node = next(node for node in ast.parse(_source).body
             if isinstance(node,ast.ClassDef) and node.name=='GlobalBatch')
_class_source = ''.join(_lines[_node.lineno-1:_node.end_lineno])
_class_source = _class_source.replace('class GlobalBatch(Product):','class NativeGlobalBatch(Product):')
_old = '''    def __init__(self, record):
        super().__init__(record)
        self.arena = TraceArena()'''
_new = '''    def __init__(self, record, shared_array, placement):
        self.circuits = {key:prepare(record['circuits'][key]) for key in ('L','R','P')}
        self.arena = StaticArena(shared_array,placement)'''
assert _class_source.count(_old)==1
_class_source = _class_source.replace(_old,_new)
_old = '    def __call__(self,a,b,out):\n        m,k = a.shape'
assert _class_source.count(_old)==1
_class_source = _class_source.replace(_old,'    def __call__(self,a,b,out):\n        self.arena.begin()\n        m,k = a.shape')
_old = '        assert not self.arena.owned and self.arena.live == 0\n        return out'
assert _class_source.count(_old)==1
_class_source = _class_source.replace(_old,'        self.arena.finish()\n        return out')
meter = fnp
exec(compile(_class_source,str(Path(__file__).resolve()),'exec'),globals())

