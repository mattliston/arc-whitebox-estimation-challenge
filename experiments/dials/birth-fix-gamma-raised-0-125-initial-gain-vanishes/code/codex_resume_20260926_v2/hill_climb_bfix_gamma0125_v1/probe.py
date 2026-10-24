"""Stdlib-only source prerequisite, not a numerical primitive test.
Root CLI: NATIVE -B probe.py BUILD_SHA primitive_result.json.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys

D=Path(__file__).resolve().parent


def pin(path):
    p=Path(path).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def main():
    assert len(sys.argv)==3
    out=Path(sys.argv[2]).resolve();assert not out.exists()
    bp=pin(D/'build.json');assert bp['sha256']==sys.argv[1]
    b=json.loads((D/'build.json').read_text())
    for key in ('parent','parent_build','candidate','builder','primitive','protocol','policy','history'):
        assert pin(b[key]['path'])==b[key]
    assert b['parent']['sha256']=='88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'
    assert b['append_only'] and b['source_inverse_exact'] and b['suffix']=='\n_BFIX_GAMMA = 0.125\n'
    # Import only the pinned stdlib source builder. Its main guard prevents writes.
    spec=importlib.util.spec_from_file_location('pinned_bfix_builder',b['builder']['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parent=Path(b['parent']['path']).read_text();child=Path(b['candidate']['path']).read_text()
    proof=module.check(parent,child);assert proof==b['source_proof']
    assert child.encode()[:-b['suffix_bytes']]==parent.encode()
    assert b['setup_counts']==dict(bound=682,lvfuse=649,psdb_fused=9)
    ast.parse(child)
    receipt=dict(passed=True,complete=True,candidate=b['candidate'],build=bp,probe=pin(__file__),
        source_inverse_exact=True,all_parent_function_asts_unchanged=True,
        source_proof=proof,source_only=True,numerical_execution=False,
        no_estimator_import=True,no_array_or_kernel_test=True,
        native_layer0_parity_still_required=True)
    with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
