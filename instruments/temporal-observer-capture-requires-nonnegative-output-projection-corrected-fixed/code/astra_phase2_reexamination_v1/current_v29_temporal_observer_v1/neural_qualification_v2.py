"""Inert metadata-only STORE weight-hash repair; the full eleven calls stay fresh."""
import hashlib
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
OWNER = HERE/'neural_qualification.py'
OWNER_SHA = '7a29034812b4efd72ed67bb269b9d7c8cec45bb7adec10e8048d4d7fcc20ac22'
CONTRACT = HERE.parent.parent/'learned_residual_observer_design/v4_feature_store/contract.py'
CONTRACT_SHA = '9744e49d3e17d92a57debe939f67a3f677f03d93250eca8308eb9a219303083c'
OLD_PINS = {
    OWNER:OWNER_SHA,
    HERE/'watch_neural_qualification.py':'53863b5236010ea0ac72d3c5cd78b78dc798f2bfe204485316b1533bf42b1878',
    HERE/'NEURAL_QUALIFICATION_PROTOCOL.md':'d058da92b1e6040b95147f6a568ea6fa63cd6bbb52b7a9eed7f2163c3a4276a0',
    HERE/'NEURAL_QUALIFICATION_WATCH_SPEC.md':'9b721e6f5dec069073f4d593f28d29cd5cefdf06763ad6c69186c2e6e035e827',
    CONTRACT:CONTRACT_SHA,
}
NEGATIVE = {
    'neural_qualification_v1':(
        'dddc16ce58bebf604e5e3160e9c7f9786d5f96054bf74367a3165501dbdbdb53',
        '5337f47306c5ee3a7426b613c5e6e5db718ffb882012e9f380a49420980ed411',
        'f9029d8cbc6ed1f4a87075f39a9f76067bb32d43dda514183b66fb86d52e79c2'),
    'neural_qualification_watch_v1':(
        '60c0c6bbf0e9a5747f1338195cc91c892abb1c1ab7d0e2e1ab79abb771cbe455',
        'bc8428547173d57f4ef8c2369e5f38f7ef7a5221bfdf3519fab0f97f3b434a15',
        '3532ce1a812352ef63a8e4934ae8173ae9dfe9aa6e67b2556161edd1c7c34a3e'),
}
_MODULE = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def owner_pins():
    return dict(OLD_PINS)


def preserve_negative(inputs, add, read, tick=None):
    """Authenticate actual six FALSE seals; never import their numerical evidence."""
    records = {}
    for folder,digests in NEGATIVE.items():
        paths = [HERE/folder/name for name in ('result.json','completion.json','finalization.json')]
        for path,digest in zip(paths,digests):
            add(inputs,path,digest,tick)
        result,end,final = map(read,paths)
        require(all(v.get('passed') is False and v.get('completed') is False for v in (result,end,final))
            and end['result_sha256'] == final['result_sha256'] == digests[0]
            and final['completion_sha256'] == digests[1], 'Original negative three-seal chain stays FALSE')
        records[folder] = result
    failed = records['neural_qualification_v1']; watched = records['neural_qualification_watch_v1']
    require(failed['graphs_completed'] == failed['retired_call_count'] == 6
        and failed['backward_calls'] == 4 and 'Same canonical fixed physical weights' in failed['error']
        and watched['retired_calls'] == watched['completed_graph_count'] == 6
        and watched['cleanup_passed'] and watched['whole_owned_tree_retired'], 'Exact old metadata failure, not reused eligibility')


def store_array_sha(value):
    """Exact contract.array_payload_sha256 dtype/shape/C-payload framing."""
    require(value.flags.c_contiguous, 'Canonical STORE weights remain C-contiguous')
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode('ascii'))
    digest.update(json.dumps(list(value.shape),separators=(',',':')).encode('ascii'))
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def rewrite(original, edits):
    changed,history = original,[]
    for old,new,count in edits:
        require(old and changed.count(old) == count, 'Exact neural-v2 metadata splice '+old[:100])
        positions=[]; cursor=0
        while True:
            at=changed.find(old,cursor)
            if at<0:
                break
            positions.append(at); cursor=at+len(old)
        pieces=[]; cursor=shift=0
        for at in positions:
            pieces.extend((changed[cursor:at],new)); history.append((at+shift,old,new))
            cursor=at+len(old); shift+=len(new)-len(old)
        pieces.append(changed[cursor:]); changed=''.join(pieces)
    inverse=changed
    for at,old,new in reversed(history):
        require(inverse[at:at+len(new)]==new, 'Exact positional metadata inverse span')
        inverse=inverse[:at]+old+inverse[at+len(new):]
    require(inverse==original, 'Whole frozen neural qualification source inverse')
    return changed


def derive_source():
    raw=OWNER.read_bytes()
    require(not OWNER.is_symlink() and hashlib.sha256(raw).hexdigest()==OWNER_SHA, 'Frozen failed neural-v1 owner')
    original=raw.decode()
    edits=[
        ("SCHEMA = 'current_v29_temporal_observer_neural_qualification_v1'",
         "SCHEMA = 'current_v29_temporal_observer_neural_qualification_v2'",1),
        ("HERE/'neural_qualification.py'","HERE/'neural_qualification_v2.py'",1),
        ("HERE/'watch_neural_qualification.py'","HERE/'watch_neural_qualification_v2.py'",1),
        ("HERE/'NEURAL_QUALIFICATION_PROTOCOL.md'","HERE/'NEURAL_QUALIFICATION_V2_PROTOCOL.md'",1),
        ("HERE/'NEURAL_QUALIFICATION_WATCH_SPEC.md'","HERE/'NEURAL_QUALIFICATION_V2_WATCH_SPEC.md'",1),
        ("HERE/'neural_qualification_v1'","HERE/'neural_qualification_v2'",2),
        ("args.watch_spec_sha256,**PINS,**EXTERNAL_PINS,",
         "args.watch_spec_sha256,**PINS,**EXTERNAL_PINS,**_repair.owner_pins(),",1),
        ("    adapter = load(HERE/'projected_model.py','qualified_projected_observer',MODEL_SHA)",
         "    _repair.preserve_negative(inputs,add,read,tick)\n"
         "    adapter = load(HERE/'projected_model.py','qualified_projected_observer',MODEL_SHA)",1),
        ("and rawsha(weights) == capture['fixture_bindings'][str(rank)]['raw_sha256']",
         "and _repair.store_array_sha(weights) == capture['fixture_bindings'][str(rank)]['raw_sha256']",1),
        ("    result = dict(schema=SCHEMA,stage='qualify',passed=False,completed=False,source_bindings={},input_bindings={},",
         "    result = dict(schema=SCHEMA,stage='qualify',source_repair_proof=_repair_proof,\n"
         "        prior_neural_v1_still_failed=True,no_partial_qualification_credit=True,\n"
         "        passed=False,completed=False,source_bindings={},input_bindings={},",1),
        ("            packet['saved_arrays']=save(f'graph_{index:04d}_arrays.npz',arrays)",
         "            if index>=6:\n"
         "                fixed_weight = admitted['capture']['fixture_bindings'][str(spec['rank'])]\n"
         "                packet['canonical_weight_hashes'] = dict(\n"
         "                    payload_sha256=packet['input_hashes']['weights'],\n"
         "                    STORE_dtype_shape_payload_sha256=fixed_weight['raw_sha256'],\n"
         "                    file_sha256=fixed_weight['sha256'])\n"
         "            packet['saved_arrays']=save(f'graph_{index:04d}_arrays.npz',arrays)",1),
        ("    require(record['roster'] == roster() and len(record['graph_records']) == 11, 'Exact fixed eleven evaluations')",
         "    require(record['source_repair_proof'] == _repair_proof\n"
         "        and record['prior_neural_v1_still_failed'] is True and record['no_partial_qualification_credit'] is True,\n"
         "        'Only explicit STORE framing repair, original failure preserved')\n"
         "    require(record['roster'] == roster() and len(record['graph_records']) == 11, 'Exact fixed eleven evaluations')",1),
        ("        path = HERE/'neural_qualification_v2'/f\"graph_{entry['graph_index']:04d}_complete.json\"",
         "        if entry['rank'] is not None:\n"
         "            fixed_weight = admitted['capture']['fixture_bindings'][str(entry['rank'])]\n"
         "            require(packet['canonical_weight_hashes'] == dict(\n"
         "                payload_sha256=packet['input_hashes']['weights'],\n"
         "                STORE_dtype_shape_payload_sha256=fixed_weight['raw_sha256'],\n"
         "                file_sha256=fixed_weight['sha256']), 'Payload-only and STORE hashes explicitly distinct')\n"
         "        path = HERE/'neural_qualification_v2'/f\"graph_{entry['graph_index']:04d}_complete.json\"",1),
    ]
    source=rewrite(original,edits)
    return source,dict(owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),STORE_contract_sha256=CONTRACT_SHA,
        metadata_hash_contract_only_repair=True,all_eleven_evaluations_fresh=True,
        numerical_equations_fixtures_gates_optimizer_and_resources_unchanged=True,
        failed_v1_seals={k:list(v) for k,v in NEGATIVE.items()})


def load_worker():
    global _MODULE
    if _MODULE is None:
        source,proof=derive_source()
        scope=dict(__name__='neural_qualification_v2_inert',__file__=str(Path(__file__).resolve()),
            _repair=types.SimpleNamespace(**dict(globals())),_repair_proof=proof)
        exec(compile(source,str(OWNER)+':STORE-framing-repair','exec'),scope)
        _MODULE=types.SimpleNamespace(**dict(scope))
    return _MODULE


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(load_worker(),name)


if __name__=='__main__':
    require('--run' in sys.argv, 'Inert; explicit --run and actual reviewed hashes required')
    raise SystemExit(load_worker().main())
