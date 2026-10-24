"""Bind actual N2 guard and canonicalize its single CLI separator, no rerun."""
from pathlib import Path
import ast,hashlib,json,difflib
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def write(name,x):
 p=D/name
 with p.open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2)+'\n')
 return pin(p)
guard=P/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
assert pin(guard)['sha256']=='314ed428e1986abf6b7c05daa5c0ea7c1b5e6ebe6472ba4a34c2ed59a80b5f76'
old=(D/'screen.py').read_text()
before="    assert value['command'] == command\n"
after="    actual_command = value['command'][1:] if value['command'][:1] == ['--'] else value['command']\n    assert actual_command == command\n"
assert old.count(before)==1;s=old.replace(before,after);ast.parse(s)
engine=write('screen_v2.py',s)
ei=read(D/'engine_inverse.json');ei['engine']=engine;ei['edits'].append(dict(before=before,after=after))
inv=s
for e in reversed(ei['edits']):assert inv.count(e['after'])==1;inv=inv.replace(e['after'],e['before'])
assert inv==Path(ei['parent']['path']).read_text()
inverse=write('engine_inverse_v2.json',ei)
diff=write('screen_v2.diff',''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='screen.py',tofile='screen_v2.py')))
protocol=read(D/'native_resource_protocol_v1.json');protocol['engine_inverse']=inverse
protocol['actual_native2_guard']=str(D/'normal_native2_guard.json')
protocol['guard_command_canonicalization']=dict(guard_source=pin(guard),rule="Remove exactly one leading literal '--', matching guard child launch; compare every remaining argument exactly.",limits_unchanged=True)
rp=write('native_resource_protocol_v2.json',protocol)
c=read(D/'native_screen.json');c['engine']=engine;c['resource_protocol']=rp
c['arms']['candidate']['blocks']['2']['guard']=str(D/'normal_native2_guard.json')
for n in ('2','8'):
 c['stages'][n]['result']=str(D/f'native{n}_assessment_v2.json')
 c['stages'][n]['assessment_guard']=str(D/f'native{n}_assess_guard_v2.json')
c['provenance'] += [pin(__file__),pin(D/'native_screen.json'),pin(D/'screen.py'),inverse,diff,rp,pin(guard)]
cp=write('native_screen_v2.json',c)
commands=read(D/'native_commands_v1.json');commands['config']=cp;commands['resource_protocol']=rp
commands['native_guards']['2']=str(D/'normal_native2_guard.json')
native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
for n in (2,8):commands['commands'][f'native{n}_assessment']=[str(native),'-B',engine['path'],cp['path'],cp['sha256'],str(n)]
cmd=write('native_commands_v2.json',commands)
print(json.dumps(dict(config=cp,engine=engine,inverse=inverse,protocol=rp,commands=cmd)))
