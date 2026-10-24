"""Fresh real-client pair with restored documented array limit; native receipts preserved."""
import hashlib
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
OWNER = HERE/'watch.py'
OWNER_SHA = '53e797e07d44a058e0b30dad77ad2dd8456c69be2680bdea37eeda18e2b1950b'
ENV_OWNER = HERE.parent/'current_comp_partial_d6_v1/production_watch_v2.py'
ENV_OWNER_SHA = '5e375679330dfdc0623140aede5a5bb3851f71588387a8e65d54e6d889f2361f'
PINS = {
    OWNER:OWNER_SHA,
    HERE/'PAIR_WATCH_SPEC.md':'2a655a495aaf19dcf94361bccec4770ee3e434c6e94bbf1759ea1eb8ed8b1973',
    ENV_OWNER:ENV_OWNER_SHA,
}
PREVIOUS = {
    'build_v1':('f4a0f937612f18de26e7ee4543036240f4e0a15c6faf8d19c648ddd2c2c32a1f',
        '66a38aeb36c93e53cdf905a2a416c5f7c6e4961041e3c7ba2294f9cb338afe19',
        '77b48651cc2b4c04915a7a36878d6e3007b40edc8740fdde08b390bafa7dc47d'),
    'native_v1':('ca7e70e1bf33eca00303df7efcc568477eb4d90d08a79c6a238e44c7deef292d',
        '64ee9befaf6f46d776775e5910a0d618bbee68ad45afbbaaf43eb2aef0d8fd77',
        'bce35054d98c8f49378403c536556d61c33b2e09e90510e2cbd5decf92cfbf59'),
    'pair_watch_v1':('50d3c9d96600d3cc6ccf82560d4b7fcc6f3bfb9bed693d2eab16449501911e36',
        '79897255e1fb90568af3237c74d668669774286cc4ad10e44a772381893bb1e3',
        '73ce756a7a948837ab14dd32cf1fbaba9b6f5f120be7e40f9c34dcec0ccf5a7a'),
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def admit_previous(sha, read, merge, result, tick):
    """Actual positive native phase plus its retired negative outer watch; no fake positive."""
    for path,digest in PINS.items():
        require(sha(path,tick) == digest, 'Exact original source and documented environment')
        result['source_bindings'][str(path)] = digest
    require("'FLOPSCOPE_MAX_ARRAY_BYTES':'4294967296'" in ENV_OWNER.read_text(),
            'Restore documented existing four-GiB server environment, not a candidate change')
    records = {}
    for folder,digests in PREVIOUS.items():
        paths = [HERE/folder/n for n in ('result.json','completion.json','finalization.json')]
        require([sha(p,tick) for p in paths] == list(digests), 'Exact prior actual three-seal chain')
        r,e,f = map(read,paths)
        require(e['result_sha256'] == f['result_sha256'] == digests[0]
                and f['completion_sha256'] == digests[1], 'Linked immutable previous seals')
        positive = folder != 'pair_watch_v1'
        require(all(v.get('passed') is positive and v.get('completed') is positive for v in (r,e,f)),
                'Previous native/build positive; previous overall watch still FALSE')
        result['input_bindings'].update({str(p):d for p,d in zip(paths,digests)})
        records[folder] = r
    old = records['pair_watch_v1']
    require(old['cleanup_passed'] and old['whole_owned_tree_retired'] and old['observational_groups_retired']
        and old['maximum_interval_s'] <= 1 and old['maximum_combined_rss_bytes'] < 8_000_000_000
        and old['maximum_swap_growth_bytes'] == 0 and len(old['native_markers']) == 2
        and old['remote_records'] == [], 'Actual two native retirements and zero remote completions')
    merge(old)
    log = HERE/'pair_watch_v1/remote.log'
    require(sha(log,tick) == 'f9a8c8c9c7cc5a814b53f5188a9dbc53db388044980641ccc1ade013afdb5e0e'
        and '1168900096 bytes exceeds 104857600 byte limit' in log.read_text(),
        'Diagnosed pre-network default array-limit failure only')
    result.update(previous_pair_watch_still_failed=True, prior_native_phase_reused=True,
        prior_native_observed_markers=old['native_markers'], fresh_native_calls=0,
        fresh_remote_calls_required=2, server_max_array_bytes=4294967296,
        prior_native_phase_resource_qualified=True)


def derive_source():
    raw = OWNER.read_bytes()
    require(not OWNER.is_symlink() and hashlib.sha256(raw).hexdigest() == OWNER_SHA,
            'Frozen complete direct supervisor')
    original = raw.decode()
    begin = original.index("        env = dict(os.environ,PYTHONUNBUFFERED='1')")
    end = original.index("        scratch = Path(tempfile.mkdtemp",begin)
    old_phase = original[begin:end]
    new_phase = """        _repair.admit_previous(sha,read,merge,result,tick)
        build,bs = chain(build_folder,60.)
        candidate_sha = build['candidate_sha256']
        builder.verify_build(bs['result_sha256'],candidate_sha,args.builder_sha256,args.build_protocol_sha256,tick)
        native,ns = chain(native_folder,300.)
        native_api.verify_native(native,tick)
        require(native['candidate_sha256'] == candidate_sha, 'Actual same built candidate and native calls')
        result.update(candidate_sha256=candidate_sha,build_seals=bs,native_seals=ns)
        env = dict(os.environ,PYTHONUNBUFFERED='1',FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
        reader,phase = None,'between-pairs'
        last_progress = time.monotonic()
"""
    edits = [
        ("CAP, RSS, PROGRESS = 600.","CAP, RSS, PROGRESS = 240.",1),
        ("TAG = 'current_comp_shared_gain_slice_pair_v1'","TAG = 'current_comp_shared_gain_slice_pair_v2'",1),
        ("source='watch.py', spec='PAIR_WATCH_SPEC.md'", "source='remote_watch_v2.py', spec='REMOTE_PAIR_V2_SPEC.md'",1),
        ("client='client.py'", "client='client_v2.py'",1),
        ("HERE/'pair_watch_v1', HERE/'native_v1'", "HERE/'pair_watch_v2', HERE/'native_v1'",1),
        ("(out,native_folder,build_folder,series,*remote_paths)","(out,series,*remote_paths)",1),
        ("schema='current_comp_shared_gain_slice_pair_watch_v1'", "schema='current_comp_shared_gain_slice_pair_watch_v2', repair_source_proof=_repair_proof",1),
        ("Whole600s and", "Whole240s and",1),
        ("HERE/'client.py'", "HERE/'client_v2.py'",2),
        (old_phase,new_phase,1),
        ("pair_timing_gate_passed=True,own_complete_bills=True,full_model_calls=4,",
         "pair_timing_gate_passed=True,own_complete_bills=True,full_model_calls=2,\n"
         "            joined_native_remote_calls=4,current_pair_qualified=True,",1),
        ("actual1s/whole600s tail", "actual1s/whole240s tail",1),
    ]
    source,history = original,[]
    for before,after,count in edits:
        require(source.count(before) == count, 'Exact remote-only supervision splice '+before[:80])
        positions=[]; cursor=0
        for _ in range(count):
            at=source.index(before,cursor); positions.append(at); cursor=at+len(before)
        for at in reversed(positions):
            source=source[:at]+after+source[at+len(before):]
            history.append((at,before,after))
    inverse=source
    for at,before,after in reversed(history):
        require(inverse[at:at+len(after)] == after, 'Exact supervisor inverse span')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    require(inverse.encode() == raw, 'Whole frozen monitor and numerical gates source inverse')
    return source,dict(owner_sha256=OWNER_SHA,whole_source_inverse_verified=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),fresh_remote_pair_only=True,
        old_negative_chain_preserved=True,original_numerical_and_timing_gates_unchanged=True,
        restored_documented_max_array_bytes=4294967296)


def execution_scope():
    return dict(__name__='gain_pair_watch_v2_inert',__file__=str(Path(__file__).resolve()),
        _repair=types.SimpleNamespace(**dict(globals())))


if __name__ == '__main__':
    require('--run' in sys.argv, 'Explicit reviewed run only')
    source,proof=derive_source(); scope=execution_scope()
    scope.update(__name__='__main__',_repair_proof=proof)
    exec(compile(source,str(OWNER)+':remote-only-v2','exec'),scope)
