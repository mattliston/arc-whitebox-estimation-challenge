"""Inert terminal-P65 VALID40 acquisition; actual native bills, no targets.

The qualified BD native_v2 loop is reversibly adapted only for admission,
passive byte observation and post-bill guards. No estimator source or arithmetic
is changed here. The final readout literal is the sole sealed fit output.
"""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD = NEW/'birth_direction_fit_v1'
OFFPY = P2/'data_gate/.venv'
BUDGET = 2**41
PROTOCOL_SHA = 'febed4bdccc7c031cb846e6d97f695536ca98d3bef10d07de48f1418a91b6a58'
FIT_SHA = '72b8b688862e93dca28715fab20d63fcab35c1d4a53d1e54bf33db6fcf880007'
FIT_WATCH_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
CAPTURE_SHA = '744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe'
COMMON_SHA = '52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
VERIFIER_SHA = '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608'
BANK = BD/'build_terminal_v1/estimator.py'
BANK_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
CONTROL = BD/'validation_terminal_v2'
CONTROL_SHA = 'ef7f4fcd8c6bc9275e434560ac5f514fd5f99d94077441d757ad34fb3a8df17d'
CONTROL_END_SHA = 'fbbe37f97f39210f9e60a9dc3eeaa2d647529cfef875f72fb31942a7e724a079'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
INITIALIZATION = 1383913786
VERSIONS = dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0')
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'Import specification')
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def pinner(pins):
    def pin(path, expected=None):
        path = Path(path).resolve()
        require('truthpost' not in path.parts and not
                (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
                'No target-array access through provenance')
        key = str(path)
        value = pins.get(key)
        if value is None:
            value = sha(path)
            pins[key] = value
        require(expected is None or value == expected,'Source/artifact pin: '+key)
        return value
    return pin


def bind(mapping, pin):
    for path,value in mapping.items():
        pin(value['path'],value['sha256']) if isinstance(value,dict) else pin(path,value)


def sealed(folder, digest, pin, completed=True):
    path = folder/'result.json'
    pin(path,digest); pin(folder/'completion.json')
    result,end = read(path),read(folder/'completion.json')
    require(result['passed'] and result['hashes_unchanged'] and end['passed']
            and end['result_sha256'] == digest,'Completed immutable PASS')
    if completed:
        require(result['completed'] and end['completed'],'Explicit completion')
    return result


def literal(source):
    tree = ast.parse(source)
    matches = [n for n in tree.body if isinstance(n,ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id == '_GNN_LITERAL']
    require(len(matches) == 1,'Unique complete P literal')
    n = matches[0]; lines = source.splitlines(keepends=True)
    begin = sum(map(len,lines[:n.lineno-1]))+n.col_offset
    end = sum(map(len,lines[:n.end_lineno-1]))+n.end_col_offset
    return begin,end,ast.literal_eval(n.value)


def f32(value):
    if isinstance(value,(list,tuple)):
        return tuple(f32(x) for x in value)
    require(type(value) in (int,float) and math.isfinite(value),'Finite literal value')
    raw = struct.pack('<f',value)
    require(math.isfinite(struct.unpack('<f',raw)[0]),'Finite deployed FP32 literal')
    return raw


def inverse_candidate(candidate):
    old,new = BANK.read_text(),candidate.read_text()
    a,b,p = literal(old); x,y,q = literal(new)
    require(new[:x]+old[a:b]+new[y:] == old,'Every non-P source byte unchanged')
    require(set(p) == set(q),'All current P fields preserved')
    require(len(q['head']) == 16 and all(len(r) == 64 for r in q['head'])
            and len(q['head_b']) == 16,'Only existing final65 coefficients')
    for key in p:
        if key in ('head','head_b'):
            require(f32(p[key][:15]) == f32(q[key][:15]),'Earlier P readouts frozen')
        else:
            require(f32(p[key]) == f32(q[key]),'Frozen P field '+key)
    f32(q['head'][15]); f32(q['head_b'][15])
    return dict(inverse_source_byte_exact=True,other_P_elements_byte_exact=True,
                frozen_cstar=True,changed_parameter_surface='head[15], head_b[15] only',
                allocated_terminal_parameters=65)


def authenticate_fit(fit_sha256, pin):
    require(isinstance(fit_sha256,str) and len(fit_sha256) == 64,'Future sealed fit SHA required')
    for path,digest in ((HERE/'PROTOCOL.md',PROTOCOL_SHA),(HERE/'fit.py',FIT_SHA),
                       (HERE/'watch_fit.py',FIT_WATCH_SHA),(HERE/'native.py',CAPTURE_SHA),
                       (HERE/'common.py',COMMON_SHA),(BD/'native_v2.py',NATIVE_SHA),
                       (BD/'qualify_native_v2.py',VERIFIER_SHA),(BANK,BANK_SHA)):
        pin(path,digest)
    fit = sealed(HERE/'fit_v1',fit_sha256,pin)
    require(fit['schema'] == 'current_d50_terminal_linear_fit_v1' and fit['parameters_changed'] == 65
            and fit['ridge'] == 1e-4 and fit['parameter_search'] is False
            and fit['validation_targets_accessed'] is False and fit['training_targets_accessed'] is True
            and fit['parent_sha256'] == BANK_SHA and fit['inverse_source_byte_exact']
            and fit['other_P_elements_byte_exact'] and fit['normal_equation_relative_residual'] <= 1e-9,
            'Sole prospectively fixed terminal fit, not a selected payload')
    require(fit['requested_seals']['source_sha256'] == FIT_SHA
            and fit['requested_seals']['protocol_sha256'] == PROTOCOL_SHA,'Exact executed fit source/protocol')
    bind(fit['source_bindings'],pin); bind(fit['artifacts'],pin)
    candidate = HERE/'fit_v1/estimator.py'
    require(Path(fit['candidate']).resolve() == candidate and
            Path(fit['payload']).resolve() == HERE/'fit_v1/terminal_payload.npz','Sole fit output paths')
    pin(candidate,fit['candidate_sha256']); pin(fit['payload'],fit['payload_sha256'])
    require(fit['artifacts'][str(candidate)] == fit['candidate_sha256'] and
            fit['artifacts'][fit['payload']] == fit['payload_sha256'],'Candidate/payload both sealed by fit')
    pretruth = HERE/'fit_v1/captures_sealed_before_targets.json'
    pin(pretruth,fit['capture_seal_sha256'])
    seal = read(pretruth)
    require(seal['target_accessed'] is False and seal['validation_targets_accessed'] is False
            and seal['capture_manifest_sha256'] == fit['requested_seals']['capture_sha256'],
            'All native TRAIN features sealed before fixed TRAIN solve')
    watch_sha = pin(HERE/'fit_watch_v1/result.json')
    watch = sealed(HERE/'fit_watch_v1',watch_sha,pin)
    require(watch['child_result_sha256'] == fit_sha256
            and watch['child_completion_sha256'] == pin(HERE/'fit_v1/completion.json')
            and watch['source_bindings'][str(HERE/'watch_fit.py')] == FIT_WATCH_SHA,
            'Completed resource-qualified sole fit')
    bind(watch['source_bindings'],pin); bind(watch['artifacts'],pin)
    proof = inverse_candidate(candidate)
    return dict(fit=fit,candidate=candidate,candidate_sha256=fit['candidate_sha256'],proof=proof,
                fit_result_sha256=fit_sha256,fit_watch_sha256=watch_sha)


def verifier(candidate, candidate_sha, pin):
    """Reuse all original ownership/resource checks; change only source eligibility."""
    pin(BD/'qualify_native_v2.py',VERIFIER_SHA)
    base = module(BD/'qualify_native_v2.py','tl_validation_qualified_verifier')
    text = (BD/'qualify_native_v2.py').read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name == 'verify_native')
    original = ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    before = ("    expected_sha = BANK_SHA if source == BANK else {HERE / 'build_zero_v1/estimator.py': ZERO_SHA,\n"
              "                       HERE / 'build_terminal_v1/estimator.py': TERMINAL_SHA}.get(source)")
    after = '    expected_sha = ELIGIBLE_SOURCES.get(source)'
    require(original.count(before) == 1,'One candidate-eligibility verifier splice')
    changed = original.replace(before,after)
    require(changed.replace(after,before) == original,'Exact verifier inverse')
    env = dict(vars(base),ELIGIBLE_SOURCES={BANK:BANK_SHA,candidate:candidate_sha})
    exec(compile(changed,str(__file__)+':qualified_verifier','exec'),env)
    return base,env['verify_native']


def control(pin):
    pin(CONTROL/'completion.json',CONTROL_END_SHA)
    old = sealed(CONTROL,CONTROL_SHA,pin,completed=False)
    base,verify = verifier(BANK,BANK_SHA,pin)
    values = verify(old,CONTROL,VALID,pin)
    require(old['candidate_sha256'] == BANK_SHA and old['initialization_flops'] == INITIALIZATION,
            'Current d50 complete40+repeat, not an older bank or estimated bill')
    return old,values


def native_inputs(candidate, args):
    require(args.protocol_sha256 == PROTOCOL_SHA and sha(__file__) == args.source_sha256,
            'Reviewed native/protocol source')
    require(not any(k in sys.modules for k in ('numpy','flopscope','torch')),'Admission before numerical imports')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and sys.flags.optimize == 0
            and all(os.environ.get(k) == '1' for k in BLAS)
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') == '0','Assertions/BLAS1/no fallback')
    require({k:importlib.metadata.version(k) for k in VERSIONS} == VERSIONS,'Qualified OFFPY versions')
    inputs = {}; pin = pinner(inputs)
    pin(__file__,args.source_sha256)
    context = authenticate_fit(args.fit_sha256,pin)
    require(candidate == context['candidate'],'Exactly the sole fixed fit candidate')
    old,outputs = control(pin)
    context.update(control=old,control_outputs=outputs)
    return inputs,context


def observed_predict(estimator, mlp, owner, helper):
    observer = helper.Observer(owner)
    require(sys.getprofile() is None,'Never replace another profiler')
    sys.setprofile(observer)
    try:
        served = estimator.predict(mlp,BUDGET)
    finally:
        sys.setprofile(None)
    return served,observer.export_bytes()


def check_call(result, trial, values, packet, helper, np, context):
    """Only after the original saved bill/reset; no head replay or target IO."""
    call = trial['call']; old = context['control']['trials'][call]
    require(trial['rank'] == old['rank'] and trial['includes_initialization'] == old['includes_initialization']
            and trial['flops'] == old['flops'] and trial['operation_count'] == old['operation_count'],
            'Actual own complete F/opcount equals corresponding current geometry/cold ownership')
    require(values.tobytes()[:15*1024*4] == context['control_outputs'][call][:15*1024*4],
            'Every earlier15 served row BYTE exact to current d50')
    domain = helper.diagnostics(np,packet)
    # Avoid retaining megabytes of repeated Python index lists across41 calls.
    domain['selection_cells'] = {key:{k:item[k] for k in
        ('count','floor','indices_sha256','mask_sha256')} for key,item in domain['selection_cells'].items()}
    result['domain_diagnostics'].append(dict(call=call,rank=trial['rank'],**domain))


def finish_native(result, context):
    require(len(result['trials']) == len(result['domain_diagnostics']) == 41
            and [t['rank'] for t in result['trials']] == list(VALID)+[VALID[0]],'Fixed40+repeat complete')
    require(len(result['artifacts']) == 205 and len(result['input_preparation']) == 41,'Original five-artifact lifecycles')
    initialization = result['trials'][0]['flops']-result['trials'][-1]['flops']
    require(initialization == INITIALIZATION,'Actual initialization once, no estimated correction')
    require(result['trials'][0]['output_bytes_sha256'] == result['trials'][-1]['output_bytes_sha256'],'Exact repeat')
    result.update(passed=True,completed=True,hashes_unchanged=True,initialization_flops=initialization,
        earlier15_rows_byte_exact=True,own_complete_F_and_opcounts_exact=True,
        all16_domains_qualified=True,all47_selectors_qualified=True,
        all_outputs_and_bills_sealed_before_scoring=True,score_measured=False)


def derived_main():
    require(sha(BD/'native_v2.py') == NATIVE_SHA,'Frozen native_v2 source')
    text = (BD/'native_v2.py').read_text()
    n = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name == 'main')
    original = ''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
    a = original.index("    parser.add_argument('--source',type=Path)\n")
    b = original.index('    assert Path(sys.prefix)',a)
    parser = ("    parser.add_argument('--run',action='store_true')\n"
              "    for name in ('source','protocol','fit'): parser.add_argument('--'+name+'-sha256')\n"
              "    args=parser.parse_args()\n"
              "    if not args.run: parser.print_help();return 0\n"
              "    args.source=HERE/'fit_v1/estimator.py'\n"
              "    args.split='validation';args.out=HERE/'validation_terminal_v1'\n"
              "    assert not args.out.exists(), 'Exclusive native40; no resume'\n")
    x = original.index('    if candidate==BANK:\n'); y = original.index('    manifest=json.loads(',x)
    edits = [
        (original[a:b],parser),
        ("    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',\n"
         "        HERE/'NATIVE_SCOPE_CLARIFICATION.md',Path(__file__),STORE/'manifest.json')}",
         '    inputs,prerequisite=native_inputs(candidate,args)'),
        (original[x:y],'    ranks=VALID\n'),
        ('        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,',
         '        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,\n'
         "        schema='current_d50_terminal_linear_native40_v1',completed=False,domain_diagnostics=[],\n"
         '        prerequisites=dict(fit_result_sha256=args.fit_sha256,fit_watch_sha256=prerequisite["fit_watch_sha256"],\n'
         '            source_inverse=prerequisite["proof"],control_result_sha256=CONTROL_SHA),\n'
         '        versions=VERSIONS,own_complete_bills=True,passive_byte_observation=True,\n'
         '        native_original_sha256=NATIVE_SHA,native_main_sha256=MAIN_SHA,\n'
         '        source_adapter_inverse_verified=True,paid_arithmetic_and_lifecycle_unchanged=True,'),
        ('                    served=estimator.predict(mlp,BUDGET)',
         '                    served,packet=observed_predict(estimator,mlp,module,CAPTURE_HELPER)'),
        ("                if call==0: first_bytes=values.tobytes()",
         '                check_call(result,trial,values,packet,CAPTURE_HELPER,np,prerequisite)\n'
         '                del packet\n'
         "                if call==0: first_bytes=values.tobytes()"),
        ('        result.update(passed=True,hashes_unchanged=True,\n'
         "                      initialization_flops=result['trials'][0]['flops']-result['trials'][-1]['flops'])",
         '        finish_native(result,prerequisite)'),
        ("        result['error']=traceback.format_exc()",
         '        result.update(passed=False,completed=False,hashes_unchanged=False,error=traceback.format_exc())'),
        ("    write(args.out/'completion.json',dict(passed=result['passed'],result_sha256=sha(args.out/'result.json')))",
         "    write(args.out/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=sha(args.out/'result.json')))"),
    ]
    changed = original
    for before,after in edits:
        require(changed.count(before) == 1,'Unique native metadata/observation splice')
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        require(inverse.count(after) == 1,'Unique native source inverse')
        inverse = inverse.replace(after,before)
    require(inverse == original,'Full main inverse: all remaining native statements unchanged')
    return changed,hashlib.sha256(original.encode()).hexdigest()


def main():
    if '--run' not in sys.argv:
        print('Inert: --run --source-sha256 SHA --protocol-sha256 SHA --fit-sha256 SHA')
        return 0
    os.chdir(NEW)
    source,digest = derived_main()
    require(sha(HERE/'native.py') == CAPTURE_SHA and sha(HERE/'common.py') == COMMON_SHA,
            'Qualified passive capture helpers')
    capture = module(HERE/'native.py','tl_validation_passive_capture')
    base = module(BD/'native_v2.py','tl_validation_preserved_native_loop')
    env = dict(vars(base),__file__=str(Path(__file__).resolve()),HERE=HERE,PARENT=NEW,
        native_inputs=native_inputs,observed_predict=observed_predict,check_call=check_call,
        finish_native=finish_native,CAPTURE_HELPER=capture,MAIN_SHA=digest,
        NATIVE_SHA=NATIVE_SHA,CONTROL_SHA=CONTROL_SHA,VERSIONS=VERSIONS)
    exec(compile(source,str(__file__)+':preserved_native_main','exec'),env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
