"""Prove minimum AP cover offline; deploy fixed maps only, never a runtime search."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

D=Path(__file__).resolve().parent
K=D.parent/'fable_oct19_kc_v1'
parent=K/'candidate_micro_copyall_ecut_eh14_astra_sci_v1.py'
tree=ast.parse(parent.read_text())
env={}
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ASTRA_SM40_RECORDS' for t in n.targets):
        env['_SMX_RECORDS']=dict(ast.literal_eval(n.value))
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='_SMX_RECORDS' for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),'<literal-record>','exec'),env)
n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_o20_smdirect_runs')
exec(compile(ast.Module(body=[n],type_ignores=[]),'<old-runs>','exec'),env)
old_runs=env['_o20_smdirect_runs']

def minimal_cover(pos):
    # Every nonempty subset of an actual output set (max7 in this source).
    assert len(pos)<=10
    ap={}
    for mask in range(1,1<<len(pos)):
        seq=tuple(pos[i] for i in range(len(pos)) if mask&(1<<i))
        if len(seq)<=2 or len({b-a for a,b in zip(seq,seq[1:])})==1:
            ap[mask]=seq
    @lru_cache(None)
    def solve(mask):
        if not mask:
            return ()
        first=mask&-mask
        candidates=[]
        for part,seq in ap.items():
            if part&first and part&mask==part:
                rest=solve(mask^part)
                candidates.append(tuple(sorted((seq,)+rest)))
        return min(candidates,key=lambda cover:(len(cover),tuple(-len(x) for x in cover),cover))
    return solve((1<<len(pos))-1)

fixed={};cases=[]
for name,record in env['_SMX_RECORDS'].items():
    for kind in ('L','R'):
        plan=record['prepared'][kind]
        out={ref:i for i,ref in enumerate(plan[2])}
        for gi,((depth,op,lit),nodes) in enumerate(plan[1]):
            pos=tuple(sorted(out[ref] for ref,args in nodes if ref in out))
            old=tuple(old_runs(pos))
            best=minimal_cover(pos)
            assert sorted(i for seq in best for i in seq)==list(pos)
            assert len({i for seq in best for i in seq})==len(pos)
            assert len(best)<=len(old)
            if len(best)<len(old):
                if pos in fixed:
                    assert fixed[pos]==best
                fixed[pos]=best
            cases.append(dict(record=name,kind=kind,group=gi,positions=pos,
                              greedy_runs=old,minimum_runs=best,improved=len(best)<len(old)))
assert len(fixed)==2
stamp=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
compact=D/'tail_smdirect_compact_internal_v1.py'
assert hashlib.sha256(compact.read_bytes()).hexdigest()=='590dd3b050bfeb1561c049ed4d4cbdd9b4089d9c45f29a9b1420a67eb3106a41'
text=compact.read_text().replace("'o20_direct_compact_v1'","'o20_direct_compact_ap_v1'")
text+=f'''
# [astra-bill {stamp}] Fixed output-index AP covers from exhaustive metadata proof.
# Output positions and per-node arithmetic are unchanged. Only independent
# same-group nodes are scheduled together into fewer basic strided slices.
# No runtime combinatorial search, new gather, output permutation, or copy.
_O20_SMCOMPACT_RUNS_PARENT = _o20_smdirect_runs
_O20_SMCOMPACT_FIXED_AP = {fixed!r}

def _o20_smdirect_runs(positions):
    positions = tuple(sorted(positions))
    fixed = _O20_SMCOMPACT_FIXED_AP.get(positions)
    if fixed is not None:
        yield from fixed
    else:
        yield from _O20_SMCOMPACT_RUNS_PARENT(positions)
'''
tail=D/'tail_smdirect_compact_ap_complete_v1.py'
assert not tail.exists()
compile(text,str(tail),'exec')
tail.write_text(text)
prior=json.loads((D/'SMDIRECT_COMPACT_PREPARED_v1.json').read_text())
for row in prior['plans']:
    matching=[x for x in cases if x['record']==row['record'] and x['kind']==row['kind']]
    saved=sum(len(x['greedy_runs'])-len(x['minimum_runs']) for x in matching)
    row['additional_fixed_output_calls_saved']=saved
    row['new_compact_arithmetic_calls']-=saved
    for x in matching:
        if x['improved']:
            plan=env['_SMX_RECORDS'][row['record']]['prepared'][row['kind']]
            op=plan[1][x['group']][0][1]
            op={'sub':'subtract','neg':'multiply','scale':'multiply'}.get(op,op)
            row['new_op_counts'][op]-=len(x['greedy_runs'])-len(x['minimum_runs'])
projection=prior['rank35_census_projection']
basis=[next(r for r in prior['plans'] if [r['record'],r['kind']]==n) for n in projection['signature_names']]
new_calls=sum(n*r['new_compact_arithmetic_calls'] for n,r in zip(projection['signature_counts'],basis))
projection['projected_compact_arithmetic_calls']=new_calls
projection['projected_saved_arithmetic_calls']=projection['old_observed_arithmetic_calls']-new_calls
prior.update(utc=stamp,tail=str(tail),tail_sha256=hashlib.sha256(tail.read_bytes()).hexdigest(),
             fixed_output_ap_covers=[dict(positions=k,runs=v) for k,v in fixed.items()],
             exhaustive_all_actual_output_sets=cases,
             offline_solver_only=True,runtime_search=False,
             cache='o20_direct_compact_ap_v1 uniquely identifies both scratch compaction and fixed AP output routing.')
dest=D/'SMDIRECT_COMPACT_AP_PREPARED_v1.json'
assert not dest.exists()
dest.write_text(json.dumps(prior,indent=2)+'\n')
print(json.dumps(dict(tail=str(tail),tail_sha256=prior['tail_sha256'],fixed_covers=prior['fixed_output_ap_covers'],
                     all_output_sets_proved=len(cases),rank35_saved_calls=projection['projected_saved_arithmetic_calls']),indent=2))
