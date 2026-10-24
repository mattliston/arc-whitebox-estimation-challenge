"""Inert exact qualified-watch adapter for ONE target-blind cold35 capture.

Uses the reviewed SHARED monitor, child lifetime, sampled-resource loop and
bounded final publication, with fixed single-call/direct-c2 metadata changes.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_shared_old384_v1/watch_probe.py'
OWNER_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'


def derive_source():
    original=OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact qualified resource/lifecycle owner')
    own_start=original.index('    own = {HERE/'); own_end=original.index('    require(all(re.fullmatch',own_start)
    preparation_start=original.index("        probe = load(HERE/'probe.py'")
    preparation_end=original.index('        env = dict(os.environ',preparation_start)
    contract_start=original.index("        require(record['candidate_sha256'] == result['candidate_sha256']")
    contract_end=original.index("        for group in ('source_bindings'",contract_start)
    edits=[
        ('CAP, RSS, PROGRESS = 300.,8_000_000_000,120.', 'CAP, RSS, PROGRESS = 180.,8_000_000_000,120.'),
        ("KEYS = ('source','worker','protocol','spec','compose','fnp','design')",
         "KEYS = ('source','worker','protocol','spec','adapter','design')"),
        (original[own_start:own_end],
         "    own = {HERE/'watch_capture.py':args.source_sha256,HERE/'capture.py':args.worker_sha256,\n"
         "        HERE/'CAPTURE_PROTOCOL.md':args.protocol_sha256,HERE/'CAPTURE_WATCH_SPEC.md':args.spec_sha256,\n"
         "        HERE/'adapter.py':args.adapter_sha256,HERE/'DESIGN.md':args.design_sha256,\n"
         "        TRANSPORT:TRANSPORT_SHA, _JCA_WATCH_OWNER:_JCA_WATCH_OWNER_SHA}\n"),
        ("HERE/'probe_v1',HERE/'probe_watch_v1'", "HERE/'capture_v1',HERE/'capture_watch_v1'"),
        ("schema='current_comp_shared_old384_native37_watch_v1'", "schema='current_comp_joint_cell_capture_watch_v1'"),
        ('retired_calls=0,ranks=[37,37]', 'retired_calls=0,ranks=[35]'),
        ("        require(not classifier(sample['rows'],scope.pop('allowed')),'No foreign numerical worker')",
         "        allowed = scope.pop('allowed')\n"
         "        require(not classifier(sample['rows'],allowed),'No foreign numerical worker')\n"
         "        require(not any(int(row['pid']) not in allowed and 'flopscope_server' in str(row.get('command',''))\n"
         "            for row in sample['rows']), 'No foreign underscore-named RPC server')"),
        ("index < 2 and item['call'] == index and item['rank'] == 37,'Exact ordered37/37'",
         "index < 1 and item['call'] == index and item['rank'] == 35,'Exact one cold35'"),
        ("record['call'] == index and record['rank'] == 37 and record['context_retired']",
         "record['call'] == index and record['rank'] == 35 and record['context_retired']"),
        ('sha256=digest,call=index,rank=37)', 'sha256=digest,call=index,rank=35)'),
        (original[preparation_start:preparation_end],
         "        probe = load(HERE/'capture.py','joint_cell_capture_metadata',args.worker_sha256)\n"
         "        from types import SimpleNamespace\n"
         "        ns = SimpleNamespace(source_sha256=args.worker_sha256,adapter_sha256=args.adapter_sha256,\n"
         "            protocol_sha256=args.protocol_sha256,watcher_sha256=args.source_sha256,\n"
         "            watch_spec_sha256=args.spec_sha256,design_sha256=args.design_sha256)\n"
         "        context,adapter,paid,backend,observer = probe.admission(ns,tick)\n"
         "        for group in ('source_bindings','input_bindings'):\n"
         "            result[group].update(context[group])\n"
         "        text,proof = adapter.derive_capture()\n"
         "        result['candidate_sha256'] = adapter.CANDIDATE_SHA\n"
         "        result['capture_source_proof'] = proof\n"
         "        del text\n"
         "        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'No monitor numerics')\n"
         "        argv = [str(OFFPY),'-B','-u',str(HERE/'capture.py'),'--run',\n"
         "            '--source-sha256',args.worker_sha256,'--watcher-sha256',args.source_sha256,\n"
         "            '--protocol-sha256',args.protocol_sha256,'--watch-spec-sha256',args.spec_sha256,\n"
         "            '--adapter-sha256',args.adapter_sha256,'--design-sha256',args.design_sha256]\n"),
        ("result['retired_calls'] == 2,'Two complete calls and exit0'", "result['retired_calls'] == 1,'One complete call and exit0'"),
        (original[contract_start:contract_end],
         "        require(record['candidate_sha256'] == result['candidate_sha256']\n"
         "            and record['capture_source_proof'] == result['capture_source_proof']\n"
         "            and record['schema'] == 'current_comp_joint_cell_capture_v1'\n"
         "            and record['full_model_calls'] == record['retired_call_count'] == 1\n"
         "            and record['ranks'] == [35] and record['initialization_flops'] == 1944970298\n"
         "            and all(record[k] is True for k in ('baseline_output_byte_exact','all16_domains_qualified',\n"
         "                'all47_selectors_qualified','all_original_operations_exact','own_complete_bills',\n"
         "                'all_capture_arrays_released','all_parameter_and_estimator_owners_retired'))\n"
         "            and record['packet_bytes'] <= 512*1024**2\n"
         "            and 0 < record['diagnostic_flops'] <= 2099703203\n"
         "            and record['trials'][0]['flops'] == 403400296797 + record['diagnostic_flops'] <= 405500000000\n"
         "            and record['truth_accessed'] is False and record['score_measured'] is False\n"
         "            and record['consumer_qualified'] is False and record['repeat_acquired'] is False,\n"
         "            'One unchanged-current-c2 paid capture, not oracle or repeat authority')\n"),
        ("if p != TRANSPORT)", "if p not in (TRANSPORT,_JCA_WATCH_OWNER))")]
    changed=original; spans=[]
    for old,new in edits:
        if changed.count(old)!=1: raise RuntimeError('Unique capture-watch splice: '+old[:100])
        p=changed.index(old); changed=changed[:p]+new+changed[p+len(old):]; spans.append((p,old,new))
    inverse=changed
    for p,old,new in reversed(spans):
        if inverse[p:p+len(new)]!=new: raise RuntimeError('Capture-watch positional inverse')
        inverse=inverse[:p]+old+inverse[p+len(new):]
    if inverse!=original: raise RuntimeError('Whole qualified watch source inverse')
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),splices=len(spans),
        actual_live_resource_loop_and_retirement_unchanged=True)


def main():
    text,proof=derive_source()
    env=dict(__name__='joint_capture_qualified_watch',__file__=str(Path(__file__).resolve()),
        _JCA_WATCH_OWNER=OWNER,_JCA_WATCH_OWNER_SHA=OWNER_SHA,WATCH_SOURCE_PROOF=proof)
    exec(compile(text,'<qualified-one-cold35-capture-watch>','exec'),env)
    return env['main']()


if __name__=='__main__': raise SystemExit(main())
