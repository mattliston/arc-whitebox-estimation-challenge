"""Cheap AST-only compatibility audit. Never imports a candidate.
Usage: python -B check_astra_compute_join_v1.py NEW_PORT_OR_STACK.py
Ports lacking NPX: verifies the existing official NPX parent check.
Stacks: proves our copied NPX body/context, active hook order and exact patch.
"""
import ast
import hashlib
from pathlib import Path
import subprocess
import sys

D=Path(__file__).resolve().parent
FRAGMENT=D/'fragment_astra_compute_v1.py'
REFERENCE=D.parent/'engineer/a62_modules/s49/s54k1tw_x/candidate.py'
PIN='2ea4441025f1edc350f863abff37e150e48972ac02db2f96f3973abc9385cf4b'
assert hashlib.sha256(FRAGMENT.read_bytes()).hexdigest()==PIN
p=Path(sys.argv[1]).resolve()
s=p.read_text(); t=ast.parse(s)
ref=REFERENCE.read_text(); rt=ast.parse(ref)

def functions(tree,name):
    return [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]

def norm(node):
    return ast.dump(node,include_attributes=False)

def assignments(tree,name):
    return [n for n in tree.body if isinstance(n,ast.Assign)
            and any(isinstance(v,ast.Name) and v.id==name for v in n.targets)]

npx=functions(t,'_npx_left')
if not npx:
    check=subprocess.run([sys.executable,'-B',str(D/'check_npx_parent_v1.py'),str(p)],check=True,text=True)
    assert not functions(t,'_astra_compute_left_v1'), 'compute patch found without its required NPX source'
    print('PASS source port: copied pair-law body matches the measured route; chain must still apply NPX before compute, then audit final stack')
    raise SystemExit(0)
assert len(npx)==1 and norm(npx[0])==norm(functions(rt,'_npx_left')[0]), 'NPX copied body changed; do not apply compute clone'
hooks=assignments(t,'_K3LB_PARENT_LEFT')
patch=functions(t,'_astra_compute_left_v1')
if not patch:
    assert isinstance(hooks[-1].value,ast.Name) and hooks[-1].value.id=='_npx_left', 'another fragment replaced the captured NPX hook'
    print('PASS source prefix: exact measured NPX body and active NPX hook; safe to append compute source, subject to native gates')
else:
    assert len(patch)==1
    ft=ast.parse(FRAGMENT.read_text())
    assert norm(patch[0])==norm(functions(ft,'_astra_compute_left_v1')[0]), 'compute function differs from measured fragment'
    fragment_lines=FRAGMENT.read_text().strip()
    assert fragment_lines in s, 'full measured fragment text not present exactly'
    before=[n for n in hooks if n.lineno<patch[0].lineno]
    after=[n for n in hooks if n.lineno>patch[0].end_lineno]
    assert before and isinstance(before[-1].value,ast.Name) and before[-1].value.id=='_npx_left', 'compute captured a changed hook'
    assert len(after)==1 and isinstance(after[0].value,ast.Name) and after[0].value.id=='_astra_compute_left_v1', 'later fragment replaced compute hook'
    active=assignments(t,'_ASTRA_COMPUTE_ON')
    assert isinstance(active[-1].value,ast.Constant) and active[-1].value.value is True, 'compute patch disabled'
    print('PASS source stack: exact measured compute/NPX bodies, correct before/after capture, active final hook, no later override')
print('Current card literals and table/helper globals remain dynamic; no training constant is frozen by this fragment.')
