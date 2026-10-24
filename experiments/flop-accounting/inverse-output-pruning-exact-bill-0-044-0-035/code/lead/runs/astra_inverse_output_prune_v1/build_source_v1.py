from pathlib import Path
import ast,hashlib,json,subprocess
D=Path(__file__).resolve().parent;O=D.parents[2];P=O/'engineer/a62_modules/s49/s56ubq1tw_x/candidate.py';F=D/'fragment_inverse_output_prune_UNVALIDATED_v1.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(P)=='c149e5800751892eb4d203341154ab2444e824a067ffaf791293a49e035ca0a3'
prefix=P.read_text();fragment=F.read_text();candidates={'parent':dict(path=str(P),sha256=sha(P),mode=-1)}
for role,mode in [('OFF',0),('ON',1)]:
 tail=fragment.replace('_ASTRA_IOP_ON = True','_ASTRA_IOP_ON = False',1) if not mode else fragment
 p=D/f'candidate_{role}_v1.py';out=prefix+'\n\n'+tail
 with p.open('x') as f:f.write(out)
 assert out.startswith(prefix+'\n\n')
 candidates[role]=dict(path=str(p),sha256=sha(p),mode=mode)
source=ast.parse(prefix);fns={}
for name in ['_ks_unwind','_ipt_side','_l3o_last','_l3o_pairs','_rs_level1','_rs_nat_perm','_rs_basis_views','_rs_out_quadrants','_hzg_product','_hz7_product','_hz_ks_product','_ubx_ks_product','_lvf_ops']:
 fns[name]=[dict(line=n.lineno,source_sha256=hashlib.sha256(ast.get_source_segment(prefix,n).encode()).hexdigest()) for n in source.body if isinstance(n,ast.FunctionDef) and n.name==name]
pins={str(p):sha(p) for p in [P,F,D/'inverse_plan_HELD_v2.py',D/'tiny_fragment_dirty_v1.py',D/'tiny_fragment_dirty_v2.py',D/'TINY_FRAGMENT_DIRTY_RESULT_v1.json',D/'TINY_FRAGMENT_DIRTY_RESULT_v2.json',D/'SOURCE_CEILING_AND_HELD_PLAN_v1.json',Path(__file__)]}
m=dict(utc=subprocess.check_output(['/bin/date','-u','+%FT%TZ'],text=True).strip(),status='HELD unvalidated exact inverse output pruning',candidates=candidates,source_pins=pins,context_function_generations=fns,scope='Current c149; exact canonical suffix, OFF differs only ON boolean; RLF inner callback interposition preserves its leaf addition, scoped half-strip product provenance; original final inverse output write retained.',cache='At most32 C-tree metadata-view plans; exact TC and Z1 object checks, no cached numeric values or output arrays; clear in own predict finally and parent PB cache lifecycle.',required_native='Fresh parent/OFF/ON136137 fixed6, all16 byte/MSE identity, exact scalar bill/operation counter credit, preserved existing stats, setup except new empty metadata cache, owned resource and cache/context retirement.',tiny='72 real-FNP OFF→ON→ON dirty same-owner cases, fused and unfused current inverse/L3o chain, retainedbytes/bill/chargedops exactly verified; no candidate/model imports in these fixtures.')
p=D/'SOURCE_MANIFEST_v1.json'
with p.open('x') as f:json.dump(m,f,indent=2);f.write('\n')
print(json.dumps({'manifest_sha256':sha(p),'fragment_sha256':sha(F),'candidates':candidates}))
