"""Exact internal-only SMDIRECT scratch compaction; source/metadata only."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import subprocess

D = Path(__file__).resolve().parent
K = D.parent/'fable_oct19_kc_v1'
parent = K/'candidate_micro_copyall_ecut_eh14_astra_sci_v1.py'
src = parent.read_text()
assert hashlib.sha256(src.encode()).hexdigest() == '09fb70c86fa245d083f1db5d9c06dcde90cd71cbc8aaeaabb1bcb134aef59ded'
tree = ast.parse(src)
defs = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
old = ast.get_source_segment(src,defs['_astra_sm40_linear_grouped'])
old_mapping = '''            internal_nodes = {i: ref for i, (ref, args) in enumerate(nodes)
                              if ref not in output_at}'''
new_mapping = '''            internal_refs = tuple(ref for ref, args in nodes if ref not in output_at)
            internal_nodes = dict(enumerate(internal_refs))
            internal_at = {ref: i for i, ref in internal_nodes.items()}'''
assert old.count(old_mapping)==1
old_value = 'else node_views[node_at + i])'
new_value = 'else node_views[node_at + internal_at[ref]])'
assert old.count(old_value)==1
new = old.replace(old_mapping,new_mapping).replace(old_value,new_value)
key = ast.get_source_segment(src,defs['_sm40_program_key'])
assert key.count("'o20_direct_v1'")==1
key = key.replace("'o20_direct_v1'","'o20_direct_compact_v1'")
stamp = subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
tail = D/'tail_smdirect_compact_internal_v1.py'
assert not tail.exists()
text = f'''# [astra-bill {stamp}] Exact SMDIRECT internal scratch compaction.
# Each group's private internal nodes occupy its existing batch prefix.
# Public LA/RB output positions, per-node operator and operand order are fixed.
# The group offset remains unchanged; no other group's scratch can be overwritten.
# Distinct key is shared by runtime and the existing metadata-only precompiler.
assert _O20_SMDIRECT_ON

''' + key + '\n\n' + new + '\n'
compile(text,str(tail),'exec')
tail.write_text(text)
env={}
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ASTRA_SM40_RECORDS' for t in n.targets):
        env['_SMX_RECORDS']=dict(ast.literal_eval(n.value))
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='_SMX_RECORDS' for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),'<literal-record>','exec'),env)
exec(compile(ast.Module(body=[defs['_o20_smdirect_runs']],type_ignores=[]),'<runs>','exec'),env)
rows=[]
opname={'sub':'subtract','scale':'multiply','neg':'multiply'}
for name,record in env['_SMX_RECORDS'].items():
    for kind in ('L','R'):
        inputs,groups,outputs,uses=record['prepared'][kind]
        output_at={ref:i for i,ref in enumerate(outputs)}
        known=set(inputs)
        old_counts,new_counts=collections.Counter(),collections.Counter()
        changes=[]
        for gi,((depth,op,lit),nodes) in enumerate(groups):
            assert all(arg in known for ref,args in nodes for arg in args)
            assert not any(ref in known for ref,args in nodes)
            internal={i:ref for i,(ref,args) in enumerate(nodes) if ref not in output_at}
            output={output_at[ref]:ref for ref,args in nodes if ref in output_at}
            ni,no=len(list(env['_o20_smdirect_runs'](internal))),len(list(env['_o20_smdirect_runs'](output)))
            nc=int(bool(internal))
            old_counts[opname.get(op,op)]+=ni+no
            new_counts[opname.get(op,op)]+=nc+no
            if ni!=nc:
                changes.append(dict(group=gi,depth=depth,op=op,nodes=len(nodes),internal_nodes=len(internal),
                                    old_internal_runs=ni,new_internal_runs=nc))
            known.update(ref for ref,args in nodes)
        assert set(outputs)<=known
        rows.append(dict(record=name,kind=kind,grouped_arithmetic_calls=len(groups),
                         old_direct_arithmetic_calls=sum(old_counts.values()),new_compact_arithmetic_calls=sum(new_counts.values()),
                         old_op_counts=dict(old_counts),new_op_counts=dict(new_counts),changed_groups=changes,
                         same_group_dag_independence_verified=True,output_positions_unchanged=True))

census_path=D/'census_copy_all_exact_35_v1.json'
warm=json.loads(census_path.read_text())['rows'][1]
actual=collections.Counter()
for row in warm['op_sites']:
    if row['function']=='_astra_sm40_linear_grouped' and row['line']==24805:
        actual[row['op']]+=row['calls']
# Three unique arithmetic-call signatures across the six actual L/R plans.
basis=[next(r for r in rows if r['record']==record and r['kind']==kind)
       for record,kind in [('signed_executable_1','L'),('signed_executable_1','R'),('signed_executable_0','L')]]
solutions=[]
for a in range(30):
    for b in range(30):
        for c in range(30):
            ns=(a,b,c)
            if all(sum(n*r['old_op_counts'].get(op,0) for n,r in zip(ns,basis))==count for op,count in actual.items()):
                solutions.append(ns)
assert solutions==[(14,10,4)]
new_calls=sum(n*r['new_compact_arithmetic_calls'] for n,r in zip(solutions[0],basis))
out=D/'SMDIRECT_COMPACT_PREPARED_v1.json'
assert not out.exists()
out.write_text(json.dumps(dict(utc=stamp,status='SOURCE_METADATA_ONLY_NO_ESTIMATOR_IMPORT',
    parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    tail=str(tail),tail_sha256=hashlib.sha256(tail.read_bytes()).hexdigest(),plans=rows,
    rank35_census_projection=dict(census=str(census_path),census_sha256=hashlib.sha256(census_path.read_bytes()).hexdigest(),
        old_observed_arithmetic_calls=sum(actual.values()),projected_compact_arithmetic_calls=new_calls,
        projected_saved_arithmetic_calls=sum(actual.values())-new_calls,
        signature_counts=solutions[0],signature_names=[[r['record'],r['kind']] for r in basis],
        unique_nonnegative_integer_solution_below30=True,flop_change=0,
        limitation='Fixed observed circuit mix projection. Production requests and wall require measurement.'),
    proof='Same-depth nodes have no mutual dependencies. All internal destinations are distinct positions inside their original group batch; downstream refs use the matching compact node view. Output LA/RB locations and node arithmetic order stay fixed. No external result exposes private scratch holes.',
    cache='o20_direct_compact_v1 distinguishes all new programs. Existing precompiler calls the same global key and intercepts insertion before arithmetic.'),indent=2)+'\n')
print(json.dumps(dict(tail=str(tail),tail_sha256=hashlib.sha256(tail.read_bytes()).hexdigest(),
                     rank35_saved_arithmetic_calls=sum(actual.values())-new_calls,flop_change=0),indent=2))
