"""External saved production/native fidelity watch; original120s/1GB loop."""
import hashlib
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_OWNER = _HERE/'watch_score_probe.py'
_OWNER_SHA = 'ce125eeb2bf774f9647b925018ba3ea1bf4e7b0a4e18277d6fab2d1c09d7d6a5'


def derive_source():
    original = _OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != _OWNER_SHA:
        raise RuntimeError('Exact completed SHARED offline supervisor source')
    a = original.index('KEYS = ('); b = original.index('\n\n\ndef require',a)
    old_keys = original[a:b]
    new_keys = "KEYS = ('source','worker','protocol','watch-spec','panel-source','panel-protocol',\n" \
        "        'panel-watch-source','panel-watch-spec','panel-score','panel-score-watch','native','native-watch','runtime')"
    a = original.index('    own = {'); b = original.index('    require(not folder.exists()',a)
    old_setup = original[a:b]
    new_setup = """    scorer = load(HERE/'fidelity.py',args.worker_sha256,'shared_fidelity_metadata_only')
    own,inputs,folder,out = scorer.watch_setup(args)
    own.update({TRANSPORT:TRANSPORT_SHA,_OWNER:_OWNER_SHA,**h.MON_PINS})
    inputs.update(h.MOCK_PINS)
"""
    a = original.index("        flags = ('integrity_passed'"); b = original.index('        union = {}\n',a)
    old_contract = original[a:b]
    new_contract = """        scorer.watch_contract(args,r,folder,sha,tick)
        require(r['requested_seals'] == {'run':True,**{k+'_sha256':v for k,v in forwarded.items()}},
            'Exact reviewed saved-fidelity source and actual receipt arguments')
"""
    edits = [
        (old_keys,new_keys),(old_setup,new_setup),
        ("schema='current_comp_shared_old384_score_probe_watch_v1'", "schema='current_comp_shared_old384_fidelity_watch_v1'"),
        ("        scorer = load(HERE/'score_probe.py',args.worker_sha256,'shared_score_metadata_only')\n", ''),
        ("str(HERE/'score_probe.py'),'--run'", "str(HERE/'fidelity.py'),'--run'"),
        (old_contract,new_contract),
        ("if p in (HERE/'score_probe.py',HERE/'watch_score_probe.py',HERE/'SCORE_PROBE_PROTOCOL.md')",
         "if p in (HERE/'fidelity.py',HERE/'watch_fidelity.py',HERE/'FIDELITY_PROTOCOL.md',HERE/'FIDELITY_WATCH_SPEC.md')"),
        ("        for path,digest in {**own,**inputs}.items():\n            require(sha(path,tick) == digest,'Own sources and explicit acquisition inputs unchanged')",
         "        for path,digest in r['truth_bindings'].items():\n"
         "            require(scorer.truth_sha(path,tick) == digest, 'Exactly four truth byte hashes, never array decoding')\n"
         "        result['truth_bytes_rehashed_without_decoding'] = True\n"
         "        for path,digest in {**own,**inputs}.items():\n"
         "            require(sha(path,tick) == digest,'Own sources and explicit acquisition inputs unchanged')"),
        ("scientific_advancement=r['information_promising'],verified_child_nontarget_files=len(union),",
         "scientific_advancement=False,production_fidelity_qualified=r['production_fidelity_qualified'],\n"
         "            production_five_qualified=r['production_five_qualified'],verified_child_nontarget_files=len(union),"),
    ]
    changed = original; spans = []
    for before,after in edits:
        if changed.count(before) != 1:
            raise RuntimeError('Unique saved-fidelity supervisor edit '+repr(before[:100]))
        start = changed.index(before)
        changed = changed[:start]+after+changed[start+len(before):]
        spans.append((start,before,after))
    inverse = changed
    for start,before,after in reversed(spans):
        if inverse[start:start+len(after)] != after:
            raise RuntimeError('Exact saved-fidelity supervisor inverse span')
        inverse = inverse[:start]+before+inverse[start+len(after):]
    if inverse != original:
        raise RuntimeError('Entire original120s/1GB sampled/reaped/publication loop restored')
    return changed,dict(owner_sha256=_OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),whole_s=120,combined_rss_bytes=1_000_000_000,
        original_sampling_reaping_finalization_unchanged=True,four_truth_byte_hashes_not_decoded=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert; --run requires actual positive scored panel and real runtime5 receipt hashes.')
    source,proof = derive_source()
    exec(compile(source,str(_OWNER)+':saved-production-native-fidelity','exec'),globals())
