"""Exclusive source/JSON preparation of the one admitted memory-blocked HK plan."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent; H2=HERE.parent
TRACE=H2/'teammate_s32_hk_residency_trace_v1'
PRIMITIVE=H2/'hill_climb_hk15_primitive_v1'
PARENT=H2/'hill_climb_exact_bundle_v1/candidate.py'
PARENT_SHA='3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7'
V1=H2/'hill_climb_hk15_integration_v1'
ORIGINAL_ARENA_ELEMENTS=391602176


def read(p): return json.loads(p.read_text())


def pin(p):
    b=p.read_bytes(); return dict(path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def source_keys(p):
    old=()
    def value(n):
        if isinstance(n,(ast.Tuple,ast.List)):return tuple(value(x) for x in n.elts)
        if isinstance(n,ast.Constant) and type(n.value) is int:return n.value
        if isinstance(n,ast.Name) and n.id=='_PB_SINGLE':return old
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add):return value(n.left)+value(n.right)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tuple':
            assert len(n.args)==1 and not n.keywords
            return tuple(value(n.args[0]))
        raise ValueError(ast.dump(n))
    for n in ast.parse(p.read_text()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_PB_SINGLE' for t in n.targets):
            old=value(n.value)
    return old


def main():
    assert pin(PARENT)['sha256']==PARENT_SHA
    for filename,sha in (
        ('prospective_union_price_v1.json','2e014b0fa100de3bcc622936f2def6c9f244a1ef79a72eab3a89a1a33a466ca6'),
        ('prospective_setup636_v1.json','cd4fea7c6bd7d18e22dcea81bce9bdf3de61e1e53c90394f2bddd2d6a89d7866')):
        assert pin(TRACE/filename)['sha256']==sha
    assert pin(PRIMITIVE/'programs.json')['sha256']=='d9d58df654b1b73a501a3e24cb31ebc6914ebfc706b2ef24f9e8455a4d729cb1'
    assert pin(V1/'fixed_plan.json')['sha256']=='46ea1f25ae5a136a8c82a3edee54480fcecef4367ec9005ea2d2791e5a671c2d'
    prior=read(V1/'fixed_plan.json')
    union=read(TRACE/'prospective_union_price_v1.json')
    setup=read(TRACE/'prospective_setup636_v1.json')
    programs=read(PRIMITIVE/'programs.json')
    records={r['name']:r for r in programs['records']}
    peaks=programs['scratch_blocks']
    used={tuple(k) for r in union['rows'] for k in r['used_positive_keys']}
    assert len(used)==84 and len(records)==6
    retained={tuple(k) for k in setup['retained_old_keys']}
    ordered_old=tuple(k for k in source_keys(PARENT) if k in retained)
    protected={tuple(k) for k in setup['protected36']}
    assert len(ordered_old)==len(set(ordered_old))==430
    assert protected.issubset(ordered_old) and len(protected)==36
    descriptors=[]; details=[]; side_keys=set(); max_arena=0; max_complete=0; descriptor_bind=0
    batch_counts=Counter(); depth_batches=Counter()
    extra_views=Counter()
    for row in union['table']:
        key=tuple(row['key'])
        if key not in used:continue
        q=row['alternative']; d=q['depth']; a,b,c=q['child']; M,K,N=q['padded']
        _,m,k,n=key; u=2**d; p=peaks[q['name']]
        one_tree=[7**d*(x//u)*(y//u) for x,y in ((a,b),(b,c),(a,c))]
        packed=15*(a*b+b*c+a*c)
        phase_scratch=p['L']*a*b+p['R']*b*c+p['P']*a*c
        scratch=max(p['L']*a*b,p['R']*b*c,p['P']*a*c)
        output_pad=M*N if (M,N)!=(m,n) else 0
        extra=packed+phase_scratch+output_pad
        fresh_elements=((M*K if (M,K)!=(m,k) else 0)+(K*N if (K,N)!=(k,n) else 0))
        P=next(P for P in (15,5,3,1)
               if P*sum(one_tree)+packed+scratch+output_pad+fresh_elements<=ORIGINAL_ARENA_ELEMENTS)
        batches=15//P
        sizes=[P*z for z in one_tree]
        side_keys.update(((0,d,P,a,b,0),(1,d,P,b,c,sizes[0]),
                         (2,d,P,a,c,sizes[0]+sizes[1])))
        required=sum(sizes)+packed+scratch+output_pad
        complete=required+fresh_elements
        fresh=4*fresh_elements
        batch_counts[P]+=1; depth_batches[(d,P)]+=1
        max_complete=max(max_complete,complete)
        batch_views=0 if P==15 else 3*batches
        descriptors.append((key,q['name'],d,(m,k,n)))
        details.append(dict(key=key,program=q['name'],depth=d,shape=(m,k,n),
            batch_size=P,batch_count=batches,batch_view_count=batch_views,
            batch_view_bind_flops=0,complete_workspace_elements=complete,
            additional_core_calls=(batches-1)*(11*d+17),
            side_sizes=sizes,descriptor_bind_flops=extra,required_arena_elements=required,
            borrowed_bytes=4*required,fresh_padding_bytes=fresh,nominal_warm_flops=q['flops'],
            descriptor_reshape_calls=3+sum(p.values())+int(bool(output_pad)),
            packed_slot_getitem_calls=45,scratch_block_views=sum(p.values())))
        extra_views['batch_getitems']+=batch_views
        extra_views['reshape_calls']+=3+sum(p.values())+int(bool(output_pad))
        extra_views['packed_slot_getitems']+=45
        extra_views['scratch_block_getitems']+=sum(p.values())
        extra_views['scratch_base_getitems']+=1
        extra_views['packed_base_getitems']+=3
        extra_views['output_pad_getitems']+=int(bool(output_pad))
        descriptor_bind+=extra;max_arena=max(max_arena,required)
    assert len(descriptors)==84 and len(side_keys)==209
    assert batch_counts=={15:68,5:16}
    assert [tuple(r['key']) for r in details]==[tuple(r['key']) for r in prior['descriptors']]
    assert [r['program'] for r in details]==[r['program'] for r in prior['descriptors']]
    assert [r['depth'] for r in details]==[r['depth'] for r in prior['descriptors']]
    assert all(r['descriptor_bind_flops']==p['descriptor_bind_flops'] and r['nominal_warm_flops']==p['nominal_warm_flops']
               for r,p in zip(details,prior['descriptors']))
    side_counts=Counter(k[0] for k in side_keys)
    old_counts=Counter(k[0] for k in ordered_old)
    assert [side_counts[k] for k in range(3)]==[61,74,74]
    caps=tuple(total-side_counts[k] for k,total in enumerate((320,352,320)))
    assert caps==(259,278,246)
    assert all(old_counts[k]<=caps[k] for k in range(3))
    side_bind=sum(P*(7**d+7*4**(d-1))*(x//2**d)*(y//2**d)
                  for _,d,P,x,y,start in side_keys)
    assert side_bind==9093654360 and max_arena==373915956 and max_complete==379027764
    assert descriptor_bind==prior['descriptor_bind_flops']==883596288
    excluded=tuple(sorted(tuple(r['key']) for r in union['table'] if tuple(r['key']) not in used))
    assert len(excluded)==4
    constants={
        '_HK_RECORDS':records,'_HK_FIXED_DESCRIPTORS':tuple(descriptors),
        '_HK_EXPECTED_SIDE_KEYS':tuple(sorted(side_keys)),
        '_HK_OLD_PREBIND':ordered_old,'_HK_PROTECTED36':tuple(sorted(protected)),
        '_HK_OLD_SIDE_CAPS':caps,'_HK_SIDE_COUNTS':(61,74,74),
        '_HK_POOL_COUNT':len(side_keys),'_HK_SIDE_BINDING_COUNT':430+len(side_keys),
        '_HK_TOTAL_BINDING_COUNT':454+len(side_keys),
        '_HK_BATCH_COUNTS':dict(batch_counts),'_HK_MAX_REQUIRED_ELEMENTS':max_arena,
        '_HK_MAX_COMPLETE_ELEMENTS':max_complete,
        '_HK_ARENA_ELEMENTS':ORIGINAL_ARENA_ELEMENTS,'_HK_SIDE_BIND_FLOPS':side_bind,
        '_HK_DESCRIPTOR_BIND_FLOPS':descriptor_bind,
        '_HK_DESCRIPTOR_BIND_VIEWS':dict(extra_views),'_HK_EXCLUDED_FOUR':excluded,
        '_HK_DESCRIPTOR_EXPECTATIONS':{tuple(r['key']):r for r in details}}
    text='# Fixed metadata only; no runtime I/O or numerical initialization.\n'
    text+='\n'.join(k+' = '+repr(v) for k,v in constants.items())+'\n'
    ast.parse(text)
    with (HERE/'constants.py').open('x') as f:f.write(text)
    result=dict(metadata_only=True,numerical_jobs=0,parent=pin(PARENT),
        constants=pin(HERE/'constants.py'),builder=pin(Path(__file__)),
        descriptor_count=84,side_count=len(side_keys),side_counts=dict(side_counts),
        old_prebind_count=430,old_counts=dict(old_counts),old_side_caps=caps,
        retained36_exact=True,old430_unchanged=True,side_bindings=430+len(side_keys),unchanged_KS=24,total_bound_side_views=454+len(side_keys),
        batch_counts=dict(batch_counts),depth_batches=[dict(depth=k[0],batch_size=k[1],descriptors=v) for k,v in sorted(depth_batches.items())],
        selection_rule='largest divisor15/5/3/1 whose sharedworkspace plus freshA/Bpadding fits original391602176 elements',
        no_shape_depth_orientation_or_table_change=True,warm_flops_unchanged_by_construction=True,
        descriptor_views_separately_reported=dict(extra_views),
        arena_elements=ORIGINAL_ARENA_ELEMENTS,arena_bytes=4*ORIGINAL_ARENA_ELEMENTS,
        max_shared_workspace_elements=max_arena,max_complete_workspace_elements=max_complete,
        actual_allocation_note='The full fixed parent arena remains allocated plus fresh padding. The complete-workspace criterion is descriptor budgeting, not an RSS allocation certificate.',
        side_bind_flops=side_bind,descriptor_bind_flops=descriptor_bind,
        total_HK_bind_flops=side_bind+descriptor_bind,
        excluded_four_parent_fallback=excluded,descriptors=details,
        limits='Static pools only; no lazy additional descriptors/sides. Old eviction caps reduced by permanent HK reservations.',
        pins=[pin(p) for p in (TRACE/'prospective_union_price_v1.json',TRACE/'prospective_setup636_v1.json',PRIMITIVE/'programs.json',V1/'fixed_plan.json')])
    with (HERE/'fixed_plan.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(plan=pin(HERE/'fixed_plan.json'),constants=pin(HERE/'constants.py'),
                         side_bind_flops=side_bind,descriptor_bind_flops=descriptor_bind,extra_views=dict(extra_views))))


if __name__=='__main__':main()
