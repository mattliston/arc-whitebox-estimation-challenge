"""Versioned native-only correction to the pre-fit _mm route; no model run."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
H2=HERE.parent

def pin(path):
    path=Path(path).resolve();data=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))

def edit(text,before,after,count=1):
    assert text.count(before)==count,(before,text.count(before),count)
    return text.replace(before,after)

def put(name,text):
    ast.parse(text)
    with (HERE/name).open('x') as f:f.write(text)

assert not (HERE/'candidate.py').exists() and not (HERE/'disabled.py').exists()
fragment=(HERE/'fragment.py').read_text()
fragment=edit(fragment,"G=(Rphysical@Rphysical.T)/p['global_scale']","G=_mm(Rphysical,Rphysical.T)/p['global_scale']")
put('fragment_mm.py',fragment)

unit=(HERE/'unit.py').read_text()
unit=edit(unit,"exec(compile((HERE/'candidate.py').read_text(),mod.__file__,'exec'),mod.__dict__);mod._cf_init()",
    "exec(compile((HERE/'candidate.py').read_text(),mod.__file__,'exec'),mod.__dict__)\n"
    "# Use the production setup, including arena and two-phase fused prebinding.\n"
    "instance=mod.Estimator()\n"
    "instance.setup(types.SimpleNamespace(width=1024,depth=16))\n"
    "mod._cf_init()\n"
    "assert mod._SCRATCH is not None and 7 in mod._IPT_REGION\n"
    "assert mod._WINOGRAD and mod._AB_ON and mod._D7SQ_ON\n"
    "gram_key=(7,1024,1024)\n"
    "fusion=[dict.get(cache,gram_key) is not None and dict.get(cache,gram_key)[3] is not None\n"
    "    for cache in (mod._IPT_A,mod._IPT_B,mod._IPT_C)]\n"
    "assert all(fusion),('Production D7 Gram keys did not all fuse during setup',fusion,mod._PB_LOG)")
unit=edit(unit,"    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:\n        D=np.asarray(mod._cf_delta(*[mod.fnp.asarray(x) for x in arrays]))",
    "    products_before=int(mod._D7SQ_STATS['products'])\n"
    "    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:\n"
    "        D=np.asarray(mod._cf_delta(*[mod.fnp.asarray(x) for x in arrays]))\n"
    "    gram_products=int(mod._D7SQ_STATS['products'])-products_before\n"
    "    assert gram_products==(1 if len(F)==1024 else 0),(label,gram_products)")
unit=edit(unit,"helper_flops=int(ctx.summary_dict()['flops_used']))",
    "helper_flops=int(ctx.summary_dict()['flops_used']),gram_depth7_products=gram_products)")
unit=edit(unit,"model=pin(HERE/'model.npz'),candidate=pin(HERE/'candidate.py'),input_ownership_passed=True)",
    "model=pin(HERE/'model.npz'),candidate=pin(HERE/'candidate.py'),input_ownership_passed=True,\n"
    "    native_amendment=pin(HERE/'native_amendment_mm.json'),native_source_seal=pin(HERE/'native_seal_mm.json'),\n"
    "    gram_route='fully metered bank _mm FP32 depth7',production_setup=True,gram_keys_fused=fusion,\n"
    "    setup_log=list(mod._PB_LOG),unchanged_relative_tolerance=2e-5)")
put('unit_mm.py',unit)

build=(HERE/'build.py').read_text()
build=edit(build,"read(HERE/'native_seal.json')","read(HERE/'native_seal_mm.json')")
build=edit(build,"fragment=(HERE/'fragment.py').read_text()","fragment=(HERE/'fragment_mm.py').read_text()")
build=edit(build,"('PROTOCOL.md','fragment.py','build.py','candidate.py','disabled.py','model.npz','normalizers.npz','model_frozen.json')",
    "('PROTOCOL.md','fragment_mm.py','build_mm.py','unit_mm.py','launch_native_mm.py',\n"
    "    'native_amendment_mm.json','native_seal_mm.json','fit_seal.json',\n"
    "    'candidate.py','disabled.py','model.npz','normalizers.npz','model_frozen.json')")
build=edit(build,"candidate=pin(HERE/'candidate.py'),disabled=pin(HERE/'disabled.py'),sources=[pin(p) for p in files])",
    "candidate=pin(HERE/'candidate.py'),disabled=pin(HERE/'disabled.py'),sources=[pin(p) for p in files],\n"
    "    native_amendment=pin(HERE/'native_amendment_mm.json'),native_source_seal=pin(HERE/'native_seal_mm.json'),\n"
    "    fit_source_seal=pin(HERE/'fit_seal.json'),gram_route='fully metered bank _mm FP32 RRt',\n"
    "    training_feature_unchanged=True,single_candidate=True)")
build=edit(build,"('unit.py','fit_result.json','build.json')","('unit_mm.py','fit_result.json','build.json','native_seal_mm.json')")
put('build_mm.py',build)

launch=(HERE/'launch_native.py').read_text()
launch=edit(launch,"('build',LIGHT,'build.py',[]),('unit',NATIVE,'unit.py',[])",
    "('build',LIGHT,'build_mm.py',[]),('unit',NATIVE,'unit_mm.py',[])")
put('launch_native_mm.py',launch)

originals=['PROTOCOL.md','seal.json','fit_seal.json','native_seal.json','fit.py','fragment.py',
    'unit.py','build.py','launch_native.py']
record=dict(
    reason='Implement the already specified bank _mm route from the pre-fit root amendment, replacing ordinary native @ before any candidate endpoint.',
    root_amendment=pin(H2/'global_pair_feature_exact_amendment_v1.json'),
    original_sources_preserved=[pin(HERE/x) for x in originals],
    fragment_change="G=(Rphysical@Rphysical.T)/p['global_scale'] -> G=_mm(Rphysical,Rphysical.T)/p['global_scale']",
    same_algebraic_feature='RRt of FP32 standardized post covariance',
    training_sources_or_labels_changed=False,training_feature_ordinary_fp32=True,
    no_refit=True,one_frozen_model=True,one_native_candidate=True,ordinary_native_endpoint_not_run=True,
    unit_setup='Production Estimator.setup(width1024,depth16), require fused A/B/C depth7 square keys; helper product counter delta1 perfullwidth case',
    fidelity='Unchanged relative L2 <=2e-5 against ordinary FP32 NumPy/Torch outputs; actual _mm rounding must pass, not presumed equal',
    complete_bill='Native _mm and entire supplier are metered; old15.026G ordinary-GEMM estimate is superseded by measured kernel bill',
    preparation='Source/AST only, no estimator, matrix, fit, or score execution')
with (HERE/'native_amendment_mm.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(dict(fragment=pin(HERE/'fragment_mm.py'),unit=pin(HERE/'unit_mm.py'),
    amendment=pin(HERE/'native_amendment_mm.json'))))
