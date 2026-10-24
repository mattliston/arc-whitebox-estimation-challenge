"""Prepare fixed D6/ordinary numerical implementations; no matrix/model jobs."""
import ast
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent

def pin(path):
    path=Path(path).resolve();b=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def edit(text,before,after,count=1):
    assert text.count(before)==count,(before,text.count(before),count)
    return text.replace(before,after)

def put(name,text):
    ast.parse(text)
    with (HERE/name).open('x') as f:f.write(text)

for e in json.loads((HERE/'seal.json').read_text())['sources']:assert pin(e['path'])==e
for e in json.loads((HERE/'native_seal_mm.json').read_text())['sources']:assert pin(e['path'])==e
originals=['candidate.py','disabled.py','build.json','candidates.json','unit_seal.json',
    'endpoint_seal.json','unit_guard.json','unit.log','model_frozen.json','model.npz',
    'normalizers.npz','fit_seal.json','native_seal_mm.json']

fragment=(HERE/'fragment.py').read_text()
fragment6=edit(fragment,'_CF_STATS = dict(calls=0)','_CF_STATS = dict(calls=0,gram_depth6=0)')
fragment6=edit(fragment6,'def _cf_delta(F,C,K,s,m):',
    "def _cf_gram_depth6(Rphysical):\n"
    "    if tuple(Rphysical.shape)==(1024,1024):\n"
    "        out=fnp.empty((1024,1024),dtype=fnp.float32)\n"
    "        _CF_STATS['gram_depth6']+=1\n"
    "        return _ipt_product_ab(Rphysical,Rphysical.T,out,6)\n"
    "    return Rphysical@Rphysical.T\n\n"
    "def _cf_delta(F,C,K,s,m):")
fragment6=edit(fragment6,"G=(Rphysical@Rphysical.T)/p['global_scale']","G=_cf_gram_depth6(Rphysical)/p['global_scale']")
put('fragment_mm6.py',fragment6)
put('fragment_ordinary.py',fragment)

for suffix in ('mm6','ordinary'):
    build=(HERE/'build_mm.py').read_text()
    for old,new in [('native_seal_mm.json','native_seal_fidelity_v2.json'),
        ('native_amendment_mm.json','native_amendment_fidelity_v2.json'),
        ('fragment_mm.py',f'fragment_{suffix}.py'),('build_mm.py',f'build_{suffix}.py'),
        ('unit_mm.py',f'unit_{suffix}.py'),('launch_native_mm.py',f'launch_native_{suffix}.py'),
        ('candidate.py',f'candidate_{suffix}.py'),('disabled.py',f'disabled_{suffix}.py'),
        ('build.json',f'build_{suffix}.json'),('unit_seal.json',f'unit_seal_{suffix}.json'),
        ('endpoint_seal.json',f'endpoint_seal_{suffix}.json'),('candidates.json',f'candidates_{suffix}.json'),
        ('evaluate.py',f'evaluate_{suffix}.py'),('assess.py',f'assess_{suffix}.py')]:
        build=build.replace(old,new)
    route='fully metered explicit depth6 IPT FP32 RRt' if suffix=='mm6' else 'fully metered ordinary fnp matmul FP32 RRt'
    build=build.replace('fully metered bank _mm FP32 RRt',route)
    build=build.replace("name='global_pair_exact'",f"name='global_pair_exact_{suffix}'")
    put(f'build_{suffix}.py',build)

    unit=(HERE/('unit_mm.py' if suffix=='mm6' else 'unit.py')).read_text()
    for old,new in [('candidate.py',f'candidate_{suffix}.py'),('build.json',f'build_{suffix}.json'),
        ('unit_seal.json',f'unit_seal_{suffix}.json'),('unit.json',f'unit_{suffix}.json'),
        ('native_amendment_mm.json','native_amendment_fidelity_v2.json'),
        ('native_seal_mm.json','native_seal_fidelity_v2.json')]:unit=unit.replace(old,new)
    if suffix=='mm6':
        unit=edit(unit,'7 in mod._IPT_REGION','6 in mod._IPT_REGION')
        unit=edit(unit,'assert mod._WINOGRAD and mod._AB_ON and mod._D7SQ_ON','assert mod._WINOGRAD and mod._AB_ON')
        unit=edit(unit,'gram_key=(7,1024,1024)','gram_key=(6,1024,1024)')
        unit=unit.replace('Production D7 Gram','Production D6 Gram')
        unit=edit(unit,"mod._D7SQ_STATS['products']","mod._CF_STATS['gram_depth6']",count=2)
        unit=edit(unit,'gram_depth7_products=','gram_depth6_products=')
        unit=edit(unit,'fully metered bank _mm FP32 depth7','fully metered explicit IPT FP32 depth6')
    else:
        unit=edit(unit,"model=pin(HERE/'model.npz'),candidate=pin(HERE/'candidate_ordinary.py'),input_ownership_passed=True)",
            "model=pin(HERE/'model.npz'),candidate=pin(HERE/'candidate_ordinary.py'),input_ownership_passed=True,\n"
            "    native_amendment=pin(HERE/'native_amendment_fidelity_v2.json'),\n"
            "    native_source_seal=pin(HERE/'native_seal_fidelity_v2.json'),\n"
            "    gram_route='fully metered ordinary fnp FP32 RRt',unchanged_relative_tolerance=2e-5)")
    put(f'unit_{suffix}.py',unit)

    evaluate=(HERE/'evaluate.py').read_text()
    for old,new in [('build.json',f'build_{suffix}.json'),('unit.json',f'unit_{suffix}.json'),
        ('unit_guard.json',f'unit_{suffix}_guard.json')]:evaluate=evaluate.replace(old,new)
    put(f'evaluate_{suffix}.py',evaluate)

    assess=(HERE/'assess.py').read_text()
    for old,new in [('build.json',f'build_{suffix}.json'),('endpoint_seal.json',f'endpoint_seal_{suffix}.json'),
        ('disabled.py',f'disabled_{suffix}.py'),('disabled_guard.json',f'disabled_{suffix}_guard.json'),
        ('disabled_qualified.json',f'disabled_qualified_{suffix}.json'),
        ('global_pair_exact_disabled_train2_v1',f'global_pair_exact_disabled_{suffix}_train2_v1')]:assess=assess.replace(old,new)
    put(f'assess_{suffix}.py',assess)

    launch=(HERE/'launch_native.py').read_text()
    for old,new in [("('build',LIGHT,'build.py'",f"('build_{suffix}',LIGHT,'build_{suffix}.py'"),
        ("('unit',NATIVE,'unit.py'",f"('unit_{suffix}',NATIVE,'unit_{suffix}.py'"),
        ("('disabled',NATIVE,'evaluate.py'",f"('disabled_{suffix}',NATIVE,'evaluate_{suffix}.py'"),
        ("('disabled_check',LIGHT,'assess.py'",f"('disabled_check_{suffix}',LIGHT,'assess_{suffix}.py'"),
        ('disabled.py',f'disabled_{suffix}.py'),
        ('global_pair_exact_disabled_train2_v1',f'global_pair_exact_disabled_{suffix}_train2_v1')]:launch=launch.replace(old,new)
    put(f'launch_native_{suffix}.py',launch)

record=dict(utc=datetime.now(timezone.utc).isoformat(),
    reason='D7 native helper failed unchanged2e-5 gate before any endpoint; implement fixed D6 fidelity repair, ordinary fallback only if D6 fails.',
    reported_D7_first_fullwidth_relative_l2=2.159030166e-5,
    originals_preserved=[pin(HERE/x) for x in originals],
    selection_rule='Try explicit paid depth6 first. If all14 fullwidth and2tiny helper fixtures pass unchanged2e-5, select D6. Otherwise select ordinary paid @ only after its same gate passes. Score exactly one selected implementation.',
    same_model=True,same_normalizers=True,same_labels=True,same_training_feature=True,
    refit=False,amplitude_change=False,layer_change=False,score_based_implementation_selection=False,
    one_endpoint=True,relative_helper_tolerance=2e-5,
    d6_setup='Production setup and all three fused(6,1024,1024)keys; explicit local _ipt_product_ab call, no global dispatcher flags changed.',
    ordinary_cost='No low-cost claim; complete paid candidate bill decides acceptance.',
    runtime_receipts='Versioned build/unit/evaluate/assess and candidate/disabled files. Generic admission must use a new direct-H2 score directory after fidelity selection.',
    preparation='Source/AST only; no estimator, matrix, fit, label, or score jobs executed by teammate.')
with (HERE/'native_amendment_fidelity_v2.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(dict(amendment=pin(HERE/'native_amendment_fidelity_v2.json'),
    fragment_mm6=pin(HERE/'fragment_mm6.py'),unit_mm6=pin(HERE/'unit_mm6.py'))))
