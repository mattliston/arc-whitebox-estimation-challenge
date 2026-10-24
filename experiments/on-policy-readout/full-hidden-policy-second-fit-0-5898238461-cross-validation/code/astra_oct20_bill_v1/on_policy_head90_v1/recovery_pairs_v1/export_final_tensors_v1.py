"""Export the one approved oldOP fixed full90 tensor correction; no model execution.

Only head, head_b and hidden_head rows1..15 receive the sealed additive float32
correction. Original b54 SCI/REL bytes are retained as the full source prefix.
No first128 stacking, new fit, truth read, runtime change or native execution.
"""
import argparse,ast,base64,hashlib,json,math,struct,subprocess,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[7]
# Find the project using the known immutable relative prefix, independent of host.
REL=Path('pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/on_policy_head90_v1/recovery_pairs_v1')
assert str(HERE).endswith(str(REL));ROOT=Path(str(HERE)[:-len(str(REL))])
OUT=HERE/'native_final_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()

def literal(text):
    tree=ast.parse(text)
    rows=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_GNN_LITERAL' for t in n.targets)]
    assert rows
    return ast.literal_eval(rows[-1]),tree

def unpack(item):
    shape,data=item;raw=zlib.decompress(base64.b64decode(data));assert len(raw)==4*math.prod(shape)
    values=struct.unpack('<%df'%(len(raw)//4),raw);assert all(math.isfinite(v) for v in values)
    return raw,values

def functions(tree):
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root-review',required=True);ap.add_argument('--root-review-sha256',required=True);args=ap.parse_args()
    review_path=Path(args.root_review);assert sha(review_path)==args.root_review_sha256
    review=read(review_path);recipe_path=HERE/'EXPORT_RECIPE_v1.json';recipe=read(recipe_path)
    assert review['successful'] and review['one_tensor_export_authorized'] and review['recipe_sha256']==sha(recipe_path)
    assert recipe['exporter_sha256']==sha(__file__)
    for name,digest in recipe['files'].items():assert sha(ROOT/name)==digest,name
    fit_path=ROOT/recipe['fitted']['path'];fit=read(fit_path);assert fit['successful'] and fit['passed'] and fit['status']=='CLEAN_FIVE_BLOCK_PROXY_PASS_NATIVE_REQUIRED'
    assert fit['cross_validation']['mean_incremental_relative_pct']<=-.15 and fit['cross_validation']['incrementally_lower']>=54 and fit['negative_blocks']>=4
    assert all(v['physical']['passed'] for v in fit['families'])
    assert fit['no_candidate_export'] and fit['no_native_launch'] and fit['no_protected_truth_read'] and fit['added_runtime_operations']==0
    seal_path=ROOT/recipe['full90_seal']['path'];seal=read(seal_path);coef=seal['coef_by_layer']
    assert seal['family']=='full90' and seal['train']==read(HERE.parent/'PREREG_v1.json')['train'] and seal['excluded_validation']==[]
    assert seal['no_validation_truth_array_yet'] and seal['explicit_train_sum_only'] and seal['truth_read_seed_ids']==seal['train']
    assert fit['final_coef_by_layer']==coef and len(coef)==15 and all(len(row)==193 for row in coef)
    assert all(math.isfinite(v) for row in coef for v in row)
    assert fit['families'][-1]['family']=='full90' and fit['families'][-1]['coefficient_seal_sha256']==sha(seal_path)
    original={role:(ROOT/item['path']).read_text() for role,item in recipe['parents'].items()}
    parent={role:literal(text)[0] for role,text in original.items()};assert parent['science']==parent['release']
    prior=parent['science'];updated=dict(prior);assert 'first_hidden_head' not in prior
    shapes={'head':(16,64),'head_b':(16,),'hidden_head':(16,128)}
    proofs={}
    for key,lo,hi in [('head',0,64),('head_b',64,65),('hidden_head',65,193)]:
        shape=prior[key][0];assert shape==shapes[key];raw,values=unpack(prior[key]);width=hi-lo
        packed=[raw[:4*width]]
        for layer in range(1,16):
            row=coef[layer-1][lo:hi]
            packed.append(b''.join(struct.pack('<f',float(values[layer*width+j])+float(row[j])) for j in range(width)))
        newraw=b''.join(packed);assert len(newraw)==len(raw) and newraw[:4*width]==raw[:4*width]
        updated[key]=(shape,base64.b64encode(zlib.compress(newraw,9)).decode())
        actual,rounded=unpack(updated[key]);assert actual==newraw
        proofs[key]=dict(shape=list(shape),unchanged_layer0=True,original_sha256=hashlib.sha256(raw).hexdigest(),updated_sha256=hashlib.sha256(newraw).hexdigest(),float32_addition_to_existing_head=True)
    assert set(updated)==set(prior)
    for key in prior:
        if key not in shapes:assert updated[key]==prior[key]
    suffix='\n# Fixed one-step TRAIN90 on-policy readout correction; tensor-only, original b54 physics.\nassert _GNN_PARAMS is None and not _E2_GNN_STATE\nassert not _FAST_RESCUE_FRAC\n_GNN_LITERAL = '+repr(updated)+'\n'
    assert not OUT.exists();OUT.mkdir()
    files={}
    for role,text in original.items():
        target=OUT/('candidate_b54_onpolicy_second90_'+role+'_v1.py')
        with target.open('x') as stream:stream.write(text+suffix)
        written=target.read_text();assert written.startswith(text)
        decoded,tree=literal(written);assert decoded==updated
        assert functions(tree)==functions(literal(text)[1])
        for key in prior:
            assert decoded[key][0]==prior[key][0]
            if key not in shapes:assert unpack(decoded[key])[0]==unpack(prior[key])[0]
        compile(written,str(target),'exec')
        files[role]=dict(path=str(target.relative_to(ROOT)),sha256=sha(target),bytes=target.stat().st_size,parent=recipe['parents'][role],entire_parent_prefix_exact=True,function_class_ast_exact=True)
    result=dict(utc=utc(),successful=True,files=files,source_recipe_sha256=sha(recipe_path),root_review_sha256=sha(review_path),exporter_sha256=sha(__file__),fitted_sha256=sha(fit_path),full90_seal_sha256=sha(seal_path),tensor_proofs=proofs,
                all_other_tensors_and_shapes_exact=True,all_layer0_tensors_exact=True,science_release_tensors_exact=True,no_first128_head_or_stacking=True,added_runtime_operations=0,
                no_model_import_or_prediction=True,no_truth_arrays_read=True,no_new_fit_or_selection=True,no_native_adjusted_gain_or_runtime_claim=True,
                paid_companion_and_own_fullstack_qualification_still_required=True,no_release_or_fire=True)
    path=OUT/'EXPORTED_v1.json'
    with path.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(successful=True,receipt_sha256=sha(path),files=files)),flush=True)
if __name__=='__main__':main()
