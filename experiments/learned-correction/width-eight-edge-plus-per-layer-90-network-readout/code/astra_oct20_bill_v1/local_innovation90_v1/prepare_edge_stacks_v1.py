"""Prepare fixed edge8 + qualified TRAIN90 head alternatives; no native dispatch."""
import ast, hashlib, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=HERE.parent
O=D.parents[2]
K=O/'lead/runs/fable_oct19_kc_v1'
V=O/'lead/runs/astra_oct20_vm_frontier_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
stamp=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
base_tail=D/'edge8_current_rank16_stack_v1/tail_exact337422_from_v4a_v1.py'
edge_tail=D/'edge_hidden_nonlinear_v1/native_hidden8_v1/tail_msg8_edge_hidden8_nonlinear_v1.py'
disable=K/'agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py'
pins={base_tail:'765f6ad8156afa85111ac0fbdb96faec64e55c17b83eccded2a4a1c92e90efb0',edge_tail:'fb7e44209120a89b58da5e89133c903ebfd904ba00548102336c2663278a23c5',disable:'be4293e44004415fb9faba1b42a26c8428a447ae226475a77a707054aac54607'}
for p,s in pins.items():assert sha(p)==s
qualified=read(HERE/'native_v1/QUALIFIED_v1.json')
assert qualified['successful'] and read(HERE/'LOCAL_SOURCE_REVIEW_v1.json')['status']=='COPIED_SOURCE_RECEIPT_REVIEW_PASS_NATIVE_REQUIRED'
out=HERE/'edge8_stacks_v1';out.mkdir(exist_ok=False)
summ=[]
for r in qualified['qualified']:
 mode=r['candidate'].removeprefix('local_innovation90_')
 folder=out/mode;folder.mkdir()
 tails=[]
 for role,suffix in [('science','sci'),('release','rel')]:
  hp=HERE/'native_v1'/r['candidate']/Path(r['outputs'][role]['path']).name
  parent=V/'msg8_loaderfix_v1'/('candidate_msg8_teacher_loaderfix_'+suffix+'_v1.py')
  assert sha(hp)==r['outputs'][role]['sha256'] and sha(parent)==r['outputs'][role]['parent_sha256']
  hs=hp.read_text();ps=parent.read_text();assert hs.startswith(ps);tails.append(hs[len(ps):])
 assert tails[0]==tails[1]
 tail=folder/('tail_local90_'+mode+'_v1.py')
 with tail.open('x') as f:f.write(tails[0])
 name='astra_o21_edge8_local90_'+mode+'_v1'
 subprocess.run(['bash',str(K/'make_stack_v1.sh'),name,str(base_tail),str(edge_tail),str(tail)],cwd=O,check=True)
 outputs={}
 for role,sp in [('science',K/('candidate_'+name+'_sci_v1.py')),('release',K/('candidate_rel_'+name+'.py'))]:
  ep=D/'edge_hidden_nonlinear_v1/native_hidden8_v1'/('candidate_msg8_edge_hidden8_nonlinear_'+role+'_v1.py')
  edgepin={'science':'627c79b74be1747e15de3e1d8bf22acb4a3838fa168e3c55c04179c0a3bdc992','release':'786c135a04ebf5b902ff533e0190b748c46f297282ae508ebcc2daa234b77180'}[role]
  assert sha(ep)==edgepin
  expected=ep.read_text()+tails[0]+'\n\n'+disable.read_text()
  actual=sp.read_text()
  assert ast.dump(ast.parse(actual))==ast.dump(ast.parse(expected)),('full AST composition',role,mode)
  assert ast.unparse(ast.parse(actual).body[-1])=='_FAST_RESCUE_FRAC = {}'
  outputs[role]=dict(path=str(sp),sha256=sha(sp),edge8_parent=str(ep),edge8_parent_sha256=sha(ep),head_parent=r['outputs'][role])
 build=dict(utc=stamp(),name=name,model=mode,outputs=outputs,head_plan_sha256=r['plan_sha256'],head_fit_sha256=r['fit_sha256'],all_functions_classes_equal_edge8=True,full_AST_equivalent_to_edge8_plus_exact_head=True,tails=[dict(path=str(t),sha256=sha(t)) for t in (base_tail,edge_tail,tail)],no_native_launch=True,no_gain_claim=True,promotion_requires_standalone_full32_and_held_pass=True)
 write(folder/'BUILD_v1.json',build);summ.append(dict(model=mode,build=str(folder/'BUILD_v1.json'),build_sha256=sha(folder/'BUILD_v1.json'),outputs=outputs))
write(out/'PREPARED_v1.json',dict(utc=stamp(),candidates=summ,no_native_launch=True,no_gain_claim=True))
print(json.dumps(dict(prepared=[r['model'] for r in summ],no_native_launch=True)))

