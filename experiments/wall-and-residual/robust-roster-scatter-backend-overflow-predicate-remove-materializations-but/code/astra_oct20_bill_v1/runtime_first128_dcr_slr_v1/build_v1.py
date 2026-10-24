"""Exact first128 paid runtime parents plus fixed DCR robust row gathering."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;P=HERE.parent/'runtime_first128_dcr_scatter_v1';K=HERE.parents[1]/'fable_oct19_kc_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P/'READY_SOURCE_v1.json')=='a7e6249048da9bae70de37980c00b1f3ffa4095a7a55ce9f91dc99b3b95fe1a6'
base=json.loads((P/'BUILD_v1.json').read_text());tail=HERE/'tail_slr_cholesky_v1.py';nogov=K/'agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py'
assert sha(nogov)=='be4293e44004415fb9faba1b42a26c8428a447ae226475a77a707054aac54607'
outputs={'first128':{}}
for role,rec in base['outputs']['first128'].items():
 parent=Path(rec['path']);assert sha(parent)==rec['sha256'];target=HERE/f'candidate_first128_smoff_npxoff_dcr_slr_{role}_v1.py'
 with target.open('xb')as f:f.write(parent.read_bytes()+b'\n'+tail.read_bytes()+b'\n'+nogov.read_bytes())
 assert target.read_bytes().startswith(parent.read_bytes());tree=ast.parse(target.read_text());assert ast.unparse(tree.body[-1])=='_FAST_RESCUE_FRAC = {}'
 outputs['first128'][role]=dict(path=str(target),sha256=sha(target),parent_path=str(parent),parent_sha256=sha(parent))
report=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),status='SOURCE_FROZEN_TINY_AND_IMPORT_PENDING_NO_NATIVE',outputs=outputs,tail_sha256=sha(tail),parent_ready_sha256=sha(P/'READY_SOURCE_v1.json'),exact_parent_prefix=True,changed_function_only='_slr_virtual_rotation_envelope',preserved_slr_overflow_predicate_and_eager_failure=True,permanent_empty_rescue=True,no_native=True,no_submission=True,limitation='SLR preserves exact predicate and eager failure via paid Cholesky; DCR preserved exactly. Actual complete-stack bill and measured runtime remain required. Hosted and local SDK both0.12.3; no version mismatch claim.')
with(HERE/'BUILD_v1.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
