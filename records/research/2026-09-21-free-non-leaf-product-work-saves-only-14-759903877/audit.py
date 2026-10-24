import resource
from fractions import Fraction
from common import *
seal_check();done('analyze');result=read(HERE/'result.json');inputs=read(HERE/'inputs.json')
assert result['checks']['passed'] and len(result['qualified_leaf_source_sites'])==4
assert sorted((x['function'],x['line']) for x in result['qualified_leaf_source_sites'])==sorted(LEAVES)
rows=[]
for spec,out in zip(inputs['rows'],result['rows']):
    data=read(spec['attribution']['path']);trace=read(spec['identity_trace']['path'])
    total=0
    for row in data['by_op_shape']:total+=row['flops']
    ceiling=0
    for row in data['outermost_mm_calls']:ceiling+=row['flops']
    outer=ceiling;retained=0;matching=0
    for row in data['by_stack']:
        if not row['stack']:continue
        f=row['stack'][0]
        if f['function'] in ('_square_product','_packed_product_cached','_read5_leaf_packets','_w5_port_product_body') and f['line'] in (1171,1873,2447,2764):
            assert (f['function'],f['line']) in LEAVES
            if '_mm' in [q['function'] for q in row['stack']]:
                ceiling-=row['flops'];retained+=row['flops'];matching+=1
    assert (total,outer,retained,ceiling)==(out['complete_cold_flops'],out['inside'],out['retained'],out['removed'])
    assert total-ceiling==out['optimistic_remaining_complete_flops']
    frac=Fraction(ceiling,total);need=1-Fraction('3e-9')/Fraction(inputs['official_adjusted_decimal'])
    assert (frac>=need)==out['passes_uniform_3e9_planning_gate']
    operands=[(c[s],c[s+'t']) for c in trace['calls'] for s in ('a','b')]
    for use_transpose,key in ((False,'operand_identity'),(True,'operand_identity_or_transpose')):
        representatives=[];counts=[]
        def eq(a,b):return all(a[q]==b[q] for q in ('shape','dtype','sha256'))
        for value,trans in operands:
            match=None
            for i,(old,oldt) in enumerate(representatives):
                if eq(value,old) or (use_transpose and eq(value,oldt)):
                    match=i;break
            if match is None:representatives.append((value,trans));counts.append(1)
            else:counts[match]+=1
        p=out[key]
        assert (len(operands),len(counts),sum(c>1 for c in counts),sum(c-1 for c in counts),max(counts))==(
            p['occurrences'],p['classes'],p['repeated_classes'],p['occurrences_after_first'],p['maximum_multiplicity'])
    rows.append(dict(rank=out['rank'],retained_stack_records=matching,integer_partition_and_gate=True,pairwise_operand_groups=True))
seal_check()
write(HERE/'audit_result.json',dict(passed=True,rows=rows,new_native_calls=0,targets_read=False,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,rows=rows),flush=True)
