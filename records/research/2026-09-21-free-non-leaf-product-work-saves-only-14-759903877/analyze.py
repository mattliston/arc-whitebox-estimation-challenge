import ast,collections,resource,time
from fractions import Fraction
from common import *

def partition(rows):
    inside=retained=outside=0;sites=collections.Counter()
    for row in rows:
        value=row['flops'];assert isinstance(value,int) and value>=0
        stack=row['stack']
        if any(s['function']=='_mm' for s in stack):
            inside+=value
            key=(stack[0]['function'],stack[0]['line'])
            if key in LEAVES:retained+=value;sites[key]+=value
        else:outside+=value
    return dict(inside=inside,retained=retained,outside=outside,removed=inside-retained,
                site_bills=[dict(function=k[0],line=k[1],flops=v) for k,v in sorted(sites.items())])

def unit():
    frame=lambda name,line:dict(function=name,line=line)
    leaf=frame('_packed_product_cached',1873);mm=frame('_mm',2550)
    rows=[dict(flops=100,stack=[leaf,mm,mm]),dict(flops=7,stack=[frame('add',1),mm]),
          dict(flops=20,stack=[leaf]),dict(flops=0,stack=[frame('empty',1),mm]),
          dict(flops=2,stack=[frame('_packed_product_cached',1872),mm]),
          dict(flops=3,stack=[frame('other',1873),mm]),dict(flops=5,stack=[])]
    p=partition(rows)
    assert (p['inside'],p['retained'],p['outside'],p['removed'])==(112,100,25,12)
    assert p['inside']+p['outside']==137
    return dict(synthetic_stack_records=len(rows),nested_mm_counted_once=True,passed=True)

def source_map():
    source=SOURCE.read_text();tree=ast.parse(source);lines=source.splitlines()
    active={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    qualified=[]
    for name,line in sorted(LEAVES):
        node=active[name]
        statements=[n for n in ast.walk(node) if isinstance(n,(ast.Return,ast.Expr)) and n.lineno==line]
        assert len(statements)==1
        expr=statements[0].value
        if isinstance(expr,ast.BinOp):
            assert isinstance(expr.op,ast.MatMult) and isinstance(expr.left,ast.Name) and isinstance(expr.right,ast.Name)
        else:
            assert isinstance(expr,ast.Call) and ast.unparse(expr.func)=='fnp.matmul'
            assert all(isinstance(x,ast.Name) for x in expr.args)
            assert all(k.arg=='out' and isinstance(k.value,ast.Name) for k in expr.keywords)
        qualified.append(dict(function=name,line=line,statement=lines[line-1].strip()))
    transforms={}
    for name in ('_packed_product_cached','_packed_read5','_read5_leaf_packets','_w5_port_product_body'):
        for n in ast.walk(active[name]):
            if not isinstance(n,ast.Call) or not isinstance(n.func,ast.Attribute):continue
            if ast.unparse(n.func.value)!='fnp' or n.func.attr not in ('add','subtract','stack','copyto'):continue
            dest=next((k.value for k in n.keywords if k.arg=='out'),None)
            if n.func.attr=='copyto':dest=n.args[0]
            if dest is None:continue
            name_of_dest=ast.unparse(dest)
            side='a' if name_of_dest.startswith('aa[') or name_of_dest=='acopies' else ('b' if name_of_dest.startswith('bb[') or name_of_dest=='bcopies' else None)
            if side:transforms[(name,n.lineno)]=side
    assert len(transforms)==44
    return qualified,transforms

def operand_counts(calls,transpose):
    counts=collections.Counter()
    for c in calls:
        for side in ('a','b'):
            x=c[side];key=(tuple(x['shape']),x['dtype'],x['sha256'])
            if transpose:
                y=c[side+'t'];key=min(key,(tuple(y['shape']),y['dtype'],y['sha256']))
            counts[key]+=1
    return dict(occurrences=sum(counts.values()),classes=len(counts),repeated_classes=sum(v>1 for v in counts.values()),
        occurrences_after_first=sum(v-1 for v in counts.values()),maximum_multiplicity=max(counts.values()),
        includes_empty_operands=True,free_cross_side_conversion_granted=True,free_transpose_granted=transpose)

started=time.monotonic();seal_check();checks=unit();qualified,transforms=source_map()
inputs=read(HERE/'inputs.json');score=Fraction(inputs['official_adjusted_decimal']);need=1-Fraction('3e-9')/score
results=[]
for spec in inputs['rows']:
    for key in ('attribution','old_result','identity_result','identity_trace'):verify(spec[key])
    data=read(spec['attribution']['path']);trace=read(spec['identity_trace']['path'])
    calls=data['outermost_mm_calls'];assert len(calls)==len(trace['calls'])==295
    for c,t in zip(calls,trace['calls']):assert c=={k:t[k] for k in c}
    p=partition(data['by_stack']);total=data['total_flops']
    assert p['inside']==data['outermost_mm_sum']==sum(c['flops'] for c in calls)==trace['outer_mm_flops']
    assert p['inside']+p['outside']==total==trace['full_flops']
    assert sum(o['flops'] for o in data['by_op_shape'])==total
    all_matmul=sum(o['flops'] for o in data['by_op_shape'] if o['op']=='matmul')
    assert 0<p['retained']<=all_matmul<=total and set((x['function'],x['line']) for x in p['site_bills'])==LEAVES
    sides=collections.Counter()
    for x in data['by_stack']:
        if not any(f['function']=='_mm' for f in x['stack']):continue
        key=(x['stack'][0]['function'],x['stack'][0]['line'])
        if key in transforms:sides[transforms[key]]+=x['flops']
    assert sum(sides.values())<=p['removed']
    frac=Fraction(p['removed'],total);floor=total-p['removed']
    results.append(dict(rank=spec['rank'],complete_cold_flops=total,outer_calls=295,**p,
        optimistic_remaining_complete_flops=floor,optimistic_saving_fraction=float(frac),
        saving_fraction_exact=[frac.numerator,frac.denominator],passes_uniform_3e9_planning_gate=frac>=need,
        input_transform_flops=dict(sides),all_native_matmul_flops=all_matmul,
        operand_identity=operand_counts(trace['calls'],False),operand_identity_or_transpose=operand_counts(trace['calls'],True)))
out=dict(completed=True,rows=results,checks=checks,qualified_leaf_source_sites=qualified,
    input_transform_source_sites=[dict(function=k[0],line=k[1],side=v) for k,v in sorted(transforms.items())],
    required_uniform_saving_fraction=float(need),required_fraction_exact=[need.numerator,need.denominator],
    both_pass_planning_gate=all(r['passes_uniform_3e9_planning_gate'] for r in results),
    optimistic_not_executable=True,new_native_calls=0,new_arrays_decoded=0,targets_read=False,candidate_created=False,
    wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE/'result.json',out)
print({k:v for k,v in out.items() if k not in ('qualified_leaf_source_sites','input_transform_source_sites')},flush=True)
