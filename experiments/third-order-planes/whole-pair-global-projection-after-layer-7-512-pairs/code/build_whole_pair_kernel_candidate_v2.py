"""Inert source-only builder for one fixed post7 complete-pair projection.

Only --build acts, after the fixed tiny certificate passes. Uses apply_patch
to create a NEW single-file candidate; never changes parent/head parameters.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
PARENT = HERE / 'candidate_response_rows_current_runtime_v1.py'
CORE = HERE / 'whole_pair_kernel_core_v1.py'
TARGET = HERE / 'candidate_whole_pair_kernel_post7_k512_v1.py'
TINY = HERE / 'whole_pair_kernel_tiny_v2.json'
PIN_PARENT = '5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172'
PIN_CORE = '86c51be6d15f8b1c8bce463a4fb649c4974a339b5bebb20b7592669075620654'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:raise RuntimeError(message)


def physical(source,node):
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def once(source,old,new):
    require(source.count(old)==1,'Expected exactly one source edit: '+old[:80])
    return source.replace(old,new,1)


def build():
    require(not TARGET.exists() and not TARGET.is_symlink(),'Never overwrite a candidate')
    initial={p.name:sha(p) for p in (PARENT,CORE,TINY,Path(__file__))}
    require(initial[PARENT.name]==PIN_PARENT and initial[CORE.name]==PIN_CORE,'Frozen source mismatch')
    tiny=json.loads(TINY.read_text())
    require(tiny['passed'] and tiny['seals_unchanged'] and tiny['failure'] is None,'Tiny prerequisite failed')
    require(tiny['seals_before']['whole_pair_kernel_core_v1.py']==PIN_CORE,'Tiny used different core')
    for name,value in tiny['seals_before'].items():
        require(sha(HERE/name)==value,'Tiny sealed input changed: '+name)
    source=PARENT.read_text();tree=ast.parse(source)
    core_source=CORE.read_text();core_tree=ast.parse(core_source)
    helpers=[node for node in core_tree.body if isinstance(node,ast.FunctionDef)]
    require(len(helpers)==6 and all(node.name.startswith('_gpk_') for node in helpers),'Unexpected core definitions')
    require(not any(isinstance(node,ast.Name) and node.id.startswith('_gpk_') for node in ast.walk(tree)),
            'Parent already uses projection namespace')
    predicts=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_predict_binary']
    require(len(predicts)==4,'Parent prediction-definition count changed')
    node=predicts[-1];old=physical(source,node)
    new=once(old,'    newborn = None\n','    newborn = None\n    cohort_counts = []\n')
    star_line='        star = (3 * rest.T - 2 * fnp.diag(fnp.diag(rest))).astype(fnp.float32)\n'
    new=once(new,star_line,star_line+
             '        cohort_counts.append(int(selected.shape[0]))\n'
             '        if _GPK_ENABLED and l == 7:\n'
             '            p, lam, star, newborn = _gpk_compress(p, lam, star, newborn, cohort_counts, live_indices, _mm)\n')
    addition='\n\n'.join(physical(core_source,h).rstrip() for h in helpers)
    addition+='\n\n_GPK_ENABLED = True\n\n'+new
    require(source.count(old)==1,'Last prediction span not uniquely identified')
    generated=source.replace(old,addition,1)
    require(generated.count(addition)==1 and generated.replace(addition,old,1)==source,
            'Inverse source transform must restore parent BYTE FOR BYTE')
    new_tree=ast.parse(generated)
    compile(generated,TARGET.name,'exec')
    require(len(new_tree.body)==len(tree.body)+7,'Expected six helpers and one bool constant only')
    filtered=[n for n in new_tree.body
              if not (isinstance(n,ast.FunctionDef) and n.name.startswith('_gpk_'))
              and not (isinstance(n,ast.Assign) and len(n.targets)==1
                       and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_GPK_ENABLED')]
    require(len(filtered)==len(tree.body),'Original source-node census changed')
    untouched=0
    for before,after in zip(tree.body,filtered):
        if before is node:require(physical(generated,after)==new,'Prediction differs from fixed insertion')
        else:
            require(physical(source,before)==physical(generated,after),'Unrelated source node changed')
            untouched+=1
    for item in ast.walk(new_tree):
        if isinstance(item,ast.Import):
            require(all(a.name in {'flopscope','flopscope.numpy'} for a in item.names),'Non-FNP import')
        if isinstance(item,ast.ImportFrom):require(item.module=='whestbench','Unexpected import-from')
        if isinstance(item,ast.Name):require(item.id not in {'np','numpy','torch'},'Forbidden numerical backend')
    require({p.name:sha(p) for p in (PARENT,CORE,TINY,Path(__file__))}==initial,'Source changed during build')
    program=shutil.which('apply_patch');require(program is not None,'apply_patch unavailable')
    patch='*** Begin Patch\n*** Add File: '+TARGET.name+'\n'
    patch+=''.join('+'+line+'\n' for line in generated.splitlines())+'*** End Patch\n'
    process=subprocess.run([program],input=patch,text=True,cwd=HERE,capture_output=True,check=False)
    require(process.returncode==0,'apply_patch failed; preserve any artifact: '+(process.stdout+process.stderr)[-1500:])
    expected=hashlib.sha256(generated.encode()).hexdigest()
    require(sha(TARGET)==expected,'Created source differs; preserve without overwriting')
    require({p.name:sha(p) for p in (PARENT,CORE,TINY,Path(__file__))}==initial,'Input changed during creation')
    result=dict(status='BUILT_SOURCE_ONLY_NOT_EXECUTED',candidate=TARGET.name,candidate_sha256=expected,
                input_sha256=initial,unchanged_original_top_level_nodes=untouched,
                changed_existing_nodes=1,added_helpers=6,added_fixed_bool=1,
                inverse_transform_byte_exact=True,compression_post_layer=7,
                retained_old_pairs=512,unchanged_newborn_layer=7,
                frozen_response_heads_all16=True,head_payloads_byte_identical=True,
                final_Q_unreachable=True,full_native_parity_accuracy_runtime='NOT_TESTED')
    receipt=HERE/'whole_pair_kernel_candidate_source_v1.json'
    with receipt.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true')
    args=parser.parse_args()
    if args.build:build()
    else:parser.print_help()
