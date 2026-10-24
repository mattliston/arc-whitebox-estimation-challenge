"""Independent14 score supervisor; exact completed SHARED resource-loop inverse."""
import hashlib
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_OWNER = _HERE/'watch_score_probe.py'
_OWNER_SHA = 'ce125eeb2bf774f9647b925018ba3ea1bf4e7b0a4e18277d6fab2d1c09d7d6a5'


def derive_source():
    original = _OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != _OWNER_SHA:
        raise RuntimeError('Completed exact external saved-score watchdog source')
    a = original.index('KEYS = (')
    b = original.index('\n\n\ndef require',a)
    old_keys = original[a:b]
    new_keys = "KEYS = ('source','worker','protocol','watch-spec','native-source','native-watch-source',\n" \
        "        'native-protocol','native-watch-spec','native','native-watch',\n" \
        "        'native40','native40-watch','score40','score40-watch')"
    a = original.index('    own = {')
    b = original.index("    require(not folder.exists()",a)
    old_setup = original[a:b]
    new_setup = """    scorer = load(HERE/'score_panel.py',args.worker_sha256,'shared_panel_score_metadata_only')
    own,inputs,folder,out = scorer.watch_setup(args)
    own.update({TRANSPORT:TRANSPORT_SHA,_OWNER:_OWNER_SHA,**h.MON_PINS})
    inputs.update(h.MOCK_PINS)
"""
    a = original.index("        flags = ('integrity_passed'")
    b = original.index('        union = {}\n',a)
    old_contract = original[a:b]
    new_contract = """        scorer.watch_contract(args,r,folder,sha,tick)
        require(r['requested_seals'] == {'run':True,**{k+'_sha256':v for k,v in forwarded.items()}},
            'Exact reviewed acquisition and scientific prerequisite arguments')
"""
    edits = [
        (old_keys,new_keys),
        ('CAP, RSS = 120.,1_000_000_000','CAP, RSS = 120.,2_000_000_000'),
        (old_setup,new_setup),
        ("schema='current_comp_shared_old384_score_probe_watch_v1'", "schema='current_comp_shared_old384_score_panel_watch_v1'"),
        ("        scorer = load(HERE/'score_probe.py',args.worker_sha256,'shared_score_metadata_only')\n", ''),
        ("str(HERE/'score_probe.py'),'--run'", "str(HERE/'score_panel.py'),'--run'"),
        (old_contract,new_contract),
        ("if p in (HERE/'score_probe.py',HERE/'watch_score_probe.py',HERE/'SCORE_PROBE_PROTOCOL.md')",
         "if p in (HERE/'score_panel.py',HERE/'watch_score_panel.py',HERE/'SCORE_PANEL_PROTOCOL.md',HERE/'SCORE_PANEL_WATCH_SPEC.md')"),
        ("        for path,digest in {**own,**inputs}.items():\n            require(sha(path,tick) == digest,'Own sources and explicit acquisition inputs unchanged')",
         "        for binding in r['truth_bindings'].values():\n"
         "            require(scorer.sha(binding['path'],tick,truth=True) == binding['sha256'],\n"
         "                'Fresh allowlisted original14 truth byte hashes; no supervisor array decoding')\n"
         "        result['truth_bytes_rehashed_without_decoding'] = True\n"
         "        for path,digest in {**own,**inputs}.items():\n"
         "            require(sha(path,tick) == digest,'Own sources and explicit acquisition inputs unchanged')"),
        ("'Zero added swap and combined owned-tree1GB'", "'Zero added swap and combined owned-tree2GB'"),
    ]
    changed = original
    spans = []
    for before,after in edits:
        if changed.count(before) != 1:
            raise RuntimeError('Unique independent14 supervisor edit '+repr(before[:100]))
        start = changed.index(before)
        changed = changed[:start]+after+changed[start+len(before):]
        spans.append((start,before,after))
    inverse = changed
    for start,before,after in reversed(spans):
        if inverse[start:start+len(after)] != after:
            raise RuntimeError('Exact supervisor inverse span')
        inverse = inverse[:start]+before+inverse[start+len(after):]
    if inverse != original:
        raise RuntimeError('Whole completed resource loop restored exactly')
    return changed,dict(owner_sha256=_OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),whole_s=120,combined_rss_bytes=2_000_000_000,
        original_sampling_reaping_finalization_unchanged=True,truth_hashes_allowlisted_not_decoded=True)


if __name__ == '__main__':
    if '--run' not in sys.argv:
        raise SystemExit('Inert; --run with reviewed sources and actual positive acquisitions required.')
    source,proof = derive_source()
    exec(compile(source,str(_OWNER)+':independent14-score-watch','exec'),globals())
