"""Full-path tiny enumeration, lossless-cache checks, and exact current comparison."""
import ast,itertools,resource,textwrap
from fractions import Fraction
from functools import lru_cache
from common import *
seal_check();done('reprice');done('verify_paths')
data=read(HERE/'reprice_result.json');old=read(D/'complete_catalog_depth_v1/reprice_result.json')
assert data['maximum_stages']==14 and data['minimum_leaf_dimension']==1
assert old['maximum_stages']==7 and old['minimum_leaf_dimension']==8
replay=read(HERE/'paths_verified.json');assert replay['passed'] and replay['result_sha256']==sha(HERE/'reprice_result.json')
inputs=read(HERE/'reprice_inputs.json');actual={r['rank']:read(r['path']) for r in inputs['censuses']}
shape_checks=0
for rank,row in actual.items():
    for call in row['outermost_mm_calls']:
        shape=(*call['left'],call['right'][1]);v=max(shape)
        if min(shape)==0:continue
        assert v<=16384
        depth=(v-1).bit_length()
        for _ in range(depth):
            next_v=(v+1)//2;assert next_v<v;v=next_v
        assert v==1 and depth<=14;shape_checks+=1
for family in data['families']:
    assert all(min(p['shape'])>=2 and p['rank']>=1 and min(p['arithmetic'])>=0 for p in family['programs'])
for depth in range(15):
    m,k,n=(16384*t for t in (3,5,7));a,b,c=m//2**depth,k//2**depth,n//2**depth
    cost=2*a*b*c-a*c
    for _ in range(depth):cost=7*cost+7*(a*b+b*c+a*c);a,b,c=2*a,2*b,2*c
    growth=Fraction(7,4)**depth
    expected=2*m*k*n*Fraction(7,8)**depth-m*n*growth+Fraction(7,3)*(m*k+k*n+m*n)*(growth-1)
    assert cost==expected

source=(HERE/'reprice.py').read_text();tree=ast.parse(source)
def get_function(text,name):
    t=ast.parse(text);return ast.get_source_segment(text,next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name))
parts=[get_function(source,n) for n in ['gemm','fringe_parts']]
pruned_pareto=get_function(source,'pareto')
full_pareto=get_function((D/'complete_catalog_depth_v1/reprice.py').read_text(),'pareto')
calculate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calculate')
node=next(n for n in calculate.body if isinstance(n,ast.FunctionDef) and n.name=='frontier')
frontier_source='@lru_cache(None)\n'+textwrap.dedent(ast.get_source_segment(source,node))
full_fixtures=list(itertools.product(range(1,5),repeat=3))+[(1,7,5),(7,1,5),(7,5,1)]
large_fixtures=[(32,32,32),(64,64,64),(64,48,96),(64,15,256),(31,65,97)]
arenas=[256,65536,1048576,1168900096]
program_sets=[(f['name'],f['programs'],full_fixtures) for f in data['families']]
base=data['families'][0]['programs']
program_sets.append(('two_program_nontrivial_fixture',[base[-1],base[0]],large_fixtures))
checks=[];nontrivial=[];abstract_checks=0
for label,programs,fixtures in program_sets:
    @lru_cache(None)
    def exhaustive(m,k,n):
        # Enumerate every schedule; no Pareto or storage pruning.
        answer=[(2*m*k*n-m*n,4*(m*k+k*n+m*n))]
        if (m,k,n)==(1,1,1):return tuple(answer)
        for p in programs:
            for peel in [False,True]:
                dims=tuple(x//d if peel else -(-x//d) for x,d in zip((m,k,n),p['shape']))
                if min(dims)<1:continue
                assert max(dims)<max(m,k,n)
                a,b,c=dims;cm,ck,cn=[x*d for x,d in zip(dims,p['shape'])]
                if peel and (cm,ck,cn)==(m,k,n):continue
                extra=(2*m*k*n-m*n)-(2*cm*ck*cn-cm*cn) if peel else 0
                fringe=0
                if peel:
                    for x,y,z in [(cm,k-ck,cn),(cm,k,n-cn),(m-cm,k,n)]:
                        if min(x,y,z)>0:fringe=max(fringe,4*(x*y+y*z+x*z))
                ca,cb,cc=p['arithmetic'];linear=ca*a*b+cb*b*c+cc*a*c
                for cost,memory in exhaustive(a,b,c):
                    answer.append((p['rank']*cost+linear+extra,max(p['rank']*memory,fringe)))
        return tuple(answer)
    ns_full=dict(lru_cache=lru_cache,programs=programs,MIN_LEAF=1)
    exec('\n\n'.join(parts+[full_pareto,frontier_source]),ns_full)
    for arena in arenas:
        ns=dict(lru_cache=lru_cache,programs=programs,MIN_LEAF=1,ARENA=arena)
        exec('\n\n'.join(parts+[pruned_pareto,frontier_source]),ns)
        for shape in fixtures:
            brute=exhaustive(*shape);full=ns_full['frontier'](*shape);compact=ns['frontier'](*shape)
            for limit in [None,0,arena//2,arena]:
                values=[cost for cost,memory in brute if limit is None or memory<=limit]
                expected=min(values) if values else None
                for front in [full,compact]:
                    values=[cost for cost,memory,path in front if limit is None or memory<=limit]
                    assert (min(values) if values else None)==expected,(label,shape,arena,limit)
                checks.append(dict(family=label,shape=shape,arena=arena,leaf_limit=limit,
                    enumerated=len(brute),full_frontier=len(full),retained=len(compact),minimum=expected))
            if len(full)>1 or min(c for c,m in brute)<brute[0][0]:
                nontrivial.append(dict(family=label,shape=shape,arena=arena,full_frontier=len(full),retained=len(compact),
                    classical=brute[0],best=min(c for c,m in brute)))
        # Explicit monotone ancestor checks on a frontier with useful discarded entries.
        options=[(100,arena//2,'a'),(90,arena+1,'b'),(80,arena+10,'c')]
        retained=ns['pareto'](options)
        assert len(retained)==2
        for rank,linear,fringe in itertools.product([1,2,7,40],[0,5,31],[0,64,arena+1]):
            for limit in [None,arena]:
                costs=[]
                for entries in [options,retained]:
                    eligible=[rank*c+linear for c,m,p in entries if limit is None or max(rank*m,fringe)<=limit]
                    costs.append(min(eligible) if eligible else None)
                assert costs[0]==costs[1];abstract_checks+=1
        ns['frontier'].cache_clear()
    ns_full['frontier'].cache_clear();exhaustive.cache_clear()
assert nontrivial
assert any(x['retained']<x['full_frontier'] for x in nontrivial)

comparison=[];changed=[];leaf_histogram={};chosen_changes=0
for before,after in zip(old['families'],data['families']):
    assert before['name']==after['name'] and before['programs']==after['programs']
    programs={p['name']:p for p in after['programs']}
    for b,a in zip(before['records'],after['records']):
        assert b['rank']==a['rank'] and b['complete_cold_flops']==a['complete_cold_flops']
        index={x['index']:x for x in b['details']}
        for entry in a['details']:
            prev=index[entry['index']]
            assert prev['shape']==entry['shape'] and prev['caller']==entry['caller'] and prev['actual_call_flops']==entry['actual_call_flops']
            for field in ['optimistic_saved','leaf_storage_limited_saved']:assert entry[field]>=prev[field]
            if any(entry[k]!=prev[k] for k in ['optimistic_saved','leaf_storage_limited_saved']):
                changed.append(dict(family=after['name'],rank=a['rank'],index=entry['index'],caller=entry['caller'],shape=entry['shape'],
                    old_optimistic_saved=prev['optimistic_saved'],new_optimistic_saved=entry['optimistic_saved'],
                    old_limited_saved=prev['leaf_storage_limited_saved'],new_limited_saved=entry['leaf_storage_limited_saved'],
                    new_optimistic=entry['optimistic_best'],new_limited=entry['leaf_storage_limited_best']))
            for field in ['optimistic_best','leaf_storage_limited_best']:
                if entry[field]!=prev[field]:chosen_changes+=1
                chosen=entry[field]
                if chosen is None or chosen[2]=='unchanged_empty_parent':continue
                shape=tuple(entry['shape'])
                for stage in chosen[2].split(' / ')[:-1]:
                    peel=stage.endswith('[peel]');p=programs[stage[:-6] if peel else stage]
                    shape=tuple(x//d if peel else -(-x//d) for x,d in zip(shape,p['shape']))
                key=f'{after["name"]}:{field}:{min(shape)}'
                leaf_histogram[key]=leaf_histogram.get(key,0)+1
        comparison.append(dict(family=after['name'],rank=a['rank'],old_fraction=b['optimistic_fraction'],new_fraction=a['optimistic_fraction'],
            old_limited_fraction=b['leaf_storage_limited_fraction'],new_limited_fraction=a['leaf_storage_limited_fraction'],
            extra_optimistic_saved=a['optimistic_saved']-b['optimistic_saved'],extra_limited_saved=a['leaf_storage_limited_saved']-b['leaf_storage_limited_saved']))

# Exact current rule, evaluated only on integer shape metadata, without import.
parent_source=PARENT.read_text();parent_tree=ast.parse(parent_source)
rule=[n for n in parent_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_rule_depth'][-1]
ns={'_COST_RULE_CACHE':{}};exec(ast.get_source_segment(parent_source,rule),ns)
tile_nodes=[n for n in parent_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_PACKED_TILE' for t in n.targets)]
tile=ast.literal_eval(tile_nodes[-1].value);witnesses=[]
for rank,row in actual.items():
    for call in row['outermost_mm_calls']:
        m,k=call['left'];n=call['right'][1]
        if not (m==1024 and 0<k<=tile and n>1024):continue
        for start in range(0,n,tile):
            width=min(tile,n-start)
            if width<=1024 or m%32 or k%32 or width%32:continue
            depth=ns['_rule_depth']('push',m,k,width)
            if depth and min(m,k,width)//2**depth<8:
                witnesses.append(dict(rank=rank,outer_shape=(m,k,n),tile_shape=(m,k,width),caller=call['caller'],
                    selected_depth=depth,leaf_shape=(m//2**depth,k//2**depth,width//2**depth),actual_outer_flops=call['flops']))
write(HERE/'audit_result.json',dict(passed=True,structural_maximum_depth=14,positive_shape_checks=shape_checks,
    all_partition_dimensions_at_least_two=True,strict_child_maximum_decrease=True,winograd_rational_identities=15,
    exhaustive_checks=checks,nontrivial_frontier_fixtures=nontrivial,abstract_monotone_ancestor_checks=abstract_checks,
    complete_positive_leaf_family=True,lossless_frontier_pruning_verified=True,all_old_call_savings_nonworsening=True,
    comparison=comparison,changed_calls=changed,changed_chosen_entries=chosen_changes,chosen_leaf_histogram=leaf_histogram,
    current_small_leaf_dispatch_witnesses=witnesses,producer=pin(HERE/'reprice.py'),result=pin(HERE/'reprice_result.json'),
    path_replay=pin(HERE/'paths_verified.json'),old_result=pin(D/'complete_catalog_depth_v1/reprice_result.json'),
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,exhaustive_checks=len(checks),changed_calls=len(changed),changed_chosen_entries=chosen_changes,
    small_leaf_native_dispatch_witnesses=len(witnesses),comparison=comparison),flush=True)
