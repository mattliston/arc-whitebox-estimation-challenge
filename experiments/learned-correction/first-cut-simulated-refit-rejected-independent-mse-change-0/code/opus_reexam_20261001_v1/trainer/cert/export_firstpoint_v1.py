"""Freeze/export one accepted exact96 point; never judge, build or submit."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

terminal_path, output = map(lambda x:Path(x).resolve(), sys.argv[1:])
terminal = json.loads(terminal_path.read_text())
assert terminal['schema'] == 'astra_cut_first_point_v1' and terminal['accepted'] is True
assert terminal['gradient_count'] == 96 and terminal['first_point_only']
assert terminal['independent_check_selection']['count'] == terminal['sizing']['count'] == 16
assert terminal['standing_params'] == terminal['trial_params'] and terminal['standing_head'] == terminal['trial_head']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin = lambda p: {'path':str(Path(p).resolve()),'sha256':sha(p)}
for item in (*terminal['source_files'].values(),terminal['standing_params'],terminal['standing_head'],
             terminal['gradient_ranks'],terminal['gradient_selection']):
    assert sha(item['path']) == item['sha256'], item['path']
output.mkdir(exist_ok=False)
for name,item in (('params.pt',terminal['standing_params']),('head_42x128.json',terminal['standing_head'])):
    with (output/name).open('xb') as stream:
        stream.write(Path(item['path']).read_bytes())
    assert sha(output/name)==item['sha256']
head = json.loads((output/'head_42x128.json').read_text())
assert len(head)==42 and all(len(row)==128 for row in head)
assert all(not any(row) for i,row in enumerate(head) if i not in {6,17,20,21,28,29})
request = dict(terminal['request'],id=terminal['tag']+'_accepted_frozen_v1',rank=651,mode='eval',
               params=str(output/'params.pt'),hu=head,keep_all=1)
with (output/'request.json').open('x') as stream: json.dump(request,stream,indent=2)
with (output/'export.log').open('x') as stream:
    result = subprocess.run([sys.executable,'-B',terminal['source_files']['exporter']['path'],
        str(output/'params.pt'),str(output/'native_base'),
        'Accepted exact96 cut-refit point '+terminal['tag']+'; generated check is selection data, untouched judging pending.'],
        stdout=stream,stderr=subprocess.STDOUT)
assert result.returncode==0, 'Exporter failed; bundle incomplete, inspect export.log'
base = output/'native_base'
dials = json.loads((base/'physics_dials.json').read_text())
assert dials['sha256']==sha(base/'candidate.py')
sidecar = dials['per_layer_sidecar']
assert sidecar['reconstruction_all16_exact'] and sidecar['arrays_roundtrip_exact']
assert sidecar['sha256']==sha(base/'per_layer_dials.json')
assert Path(sidecar['path']).resolve()==(base/'per_layer_dials.json').resolve()
compile((base/'candidate.py').read_text(),str(base/'candidate.py'),'exec')
artifacts = {name:pin(path) for name,path in dict(params=output/'params.pt',head=output/'head_42x128.json',
    request=output/'request.json',candidate=base/'candidate.py',physics_dials=base/'physics_dials.json',
    per_layer_dials=base/'per_layer_dials.json',accepted_point=terminal_path).items()}
receipt = dict(schema='accepted_cut_point_export_v1',utc=subprocess.check_output(['date','-u','+%FT%TZ'],text=True).strip(),
    tag=terminal['tag'],artifacts=artifacts,request=terminal['request'],source_files=terminal['source_files'],
    sizing=terminal['sizing'],independent_check_selection=terminal['independent_check_selection'],
    generated_untouched_judge_reserved=terminal['generated_untouched_judge_reserved'],
    exported=True,qualified=False,native_build_complete=False,untouched_judging_complete=False,
    native_DLY_converter_required='engineer/a52_pld/dly_from_trainer_v2.py; all16 absolute values, canonical layer1 bases',
    head_rows=[6,17,20,21,28,29])
with (output/'INPUT_MANIFEST.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(dict(tag=terminal['tag'],manifest=str(output/'INPUT_MANIFEST.json'),qualified=False)))
