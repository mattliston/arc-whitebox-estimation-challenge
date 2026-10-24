"""AST-only reuse of preserved definitions; never run their module drivers."""
import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import prod
from common import A
def definitions(path,names):
    s=path.read_text();lines=s.splitlines(keepends=True)
    nodes={n.name:n for n in ast.parse(s).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    return '\n'.join(''.join(lines[nodes[name].lineno-1:nodes[name].end_lineno]) for name in names)
def environment(api=None):
    env=dict(Counter=Counter,dataclass=dataclass,Fraction=Fraction,prod=prod,__name__=__name__)
    old=A/'audit_smirnov_global_batch_static_v1.py'
    s=definitions(old,['Symbol','Meter'])
    assert s.count("self.charge('zeros', prod(shape))")==1
    s=s.replace("self.charge('zeros', prod(shape))","self.charge('zeros', 0)")
    exec(s,env)
    meter=env['Meter']();env.update(meter=meter,fnp=meter if api is None else api)
    exec(definitions(A/'streamed_leaf_winograd_v1.py',['streamed_leaf']),env)
    exec(definitions(A/'smirnov_stream8_native_v1.py',['Block','Arena','prepare','linear','Product']),env)
    exec(definitions(old,['TraceArena','GlobalBatch','place_largest_first']),env)
    return env
