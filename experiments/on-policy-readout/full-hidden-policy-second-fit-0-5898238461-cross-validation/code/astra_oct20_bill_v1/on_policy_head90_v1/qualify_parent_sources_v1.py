"""Full-source imports and official GNN arithmetic for frozen six diagnostic parents.
Uses the already qualified complete15-layer cold/warm official arithmetic fixture.
Does not launch native captures or open truth arrays.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[k] = '1'
import hashlib, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path('/home/arcworker/arc-workspace')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
manifest = HERE/'PARENT_SOURCES_v1.json'
assert sha(manifest) == 'd6eaeb233f5f8109247d46cc68672848fb10477bd18a4076829fd267a4613136'
parents = read(manifest)
final = read(HERE/'PARENT_BUILD_FINAL_v1.json')
assert final['successful'] and final['source_manifest_sha256'] == sha(manifest)
gp = Path(final['guard_path'])
assert sha(gp) == final['guard_sha256']
g = read(gp)
assert g['successful'] and g['whole_group_retired'] and g['child_exit'] == 0 and not g['cleanup_errors']
assert g['max_owned_swap_bytes'] == 0 and g['guard_sha256'] == '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
recipe = read(HERE/'PARENT_QUALIFY_RECIPE_v1.json')
assert sha(__file__) == recipe['qualifier_sha256']
imp = HERE/'check_parent_sources_import_v1.py'
tiny = HERE.parent/'hidden_complement90_v1/check_hidden_complement_tiny_v1.py'
assert sha(imp) == recipe['import_sha256'] and sha(tiny) == recipe['official_tiny_sha256']
out = HERE/'parent_qualification_v1'
out.mkdir(exist_ok=False)
with (out/'NUMPY_FREE_IMPORT_v1.log').open('x') as log:
    subprocess.run(['/home/arcworker/arc-envs/client/bin/python', '-B', str(imp), '--out', str(out/'NUMPY_FREE_IMPORT_v1.json')],
                   stdout=log, stderr=subprocess.STDOUT, check=True)
assert read(out/'NUMPY_FREE_IMPORT_v1.json')['successful']
rows = []
for family in parents['families']:
    build = ROOT/family['build_path']
    assert sha(build) == family['build_sha256']
    target = out/('TINY_'+family['key']+'_v1.json')
    with (out/('TINY_'+family['key']+'_v1.log')).open('x') as log:
        subprocess.run(['/home/arcworker/arc-envs/server/bin/python', '-B', str(tiny), '--build', str(build), '--out', str(target)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(target)
    assert report['successful'] and report['state_path_byte_identical'] and report['all15_gnn_calls_complete']
    assert report['source_sha256'] == family['science']['sha256']
    rows.append(dict(family=family['key'], source_sha256=family['science']['sha256'], tiny_path=str(target),
                     tiny_sha256=sha(target), arithmetic=report['rows']))
    print(json.dumps(dict(utc=utc(),family=family['key'],official_arithmetic_passed=True)), flush=True)
baseline = rows[-1]['arithmetic']
for row in rows:
    assert row['arithmetic'] == baseline
    row.pop('arithmetic')
result = dict(utc=utc(), successful=True, source_manifest_sha256=sha(manifest), build_guard_sha256=sha(gp),
              source_import_sha256=sha(out/'NUMPY_FREE_IMPORT_v1.json'), cases=rows,
              all_six_official_cold_warm_arithmetic_identical=True, added_flops_vs_b54=0,
              added_charged_calls_vs_b54=0, no_truth_arrays_read=True, no_native_capture_launched=True,
              root_review_required_before_capture=True, script_sha256=sha(__file__), recipe_sha256=sha(HERE/'PARENT_QUALIFY_RECIPE_v1.json'))
with (out/'QUALIFIED_v1.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(utc=utc(),successful=True,all_six_arithmetic_identical=True)), flush=True)
