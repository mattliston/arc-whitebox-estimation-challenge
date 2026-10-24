"""Inert additive endpoint2 scorer: same-history bank/null qualification.

The frozen v1 scorer owns every unchanged lifecycle, target and objective
statement.  This adapter replaces only endpoint paths/source identities and
the mismatched two-call null comparator.  It inverse-restores the entire
original source before use.  No model, fit or array work occurs by default.
"""
import hashlib
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE/'judge.py'
ORIGINAL_SHA = '443c1e7ddcc4e160be248e2a47d5f06b342013ca52037d001e63fbba44a19f6a'
NATIVE_V1_SHA = 'ebab249a8a1bdd0c2017a1a24736c98eac708310865f50588d7ab4043170e2d4'
ENDPOINT_SHA = '57d0f5e464d0c2f4313336426c0c5cde0ab0d95442305c00ad5c5b9c2a7e4021'
# Independently source-reviewed endpoint2 supervisor; no scoring was run here.
WATCH_V2_SHA = 'adfcad6c846f9fac3bb5317d4a4ef9dd7f9c8728c929aa64c5f2060358859f04'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def derived_source():
    require(sha(ORIGINAL) == ORIGINAL_SHA, 'Frozen complete endpoint1 scorer')
    require(sha(HERE/'native.py') == NATIVE_V1_SHA, 'Frozen endpoint1 native owner')
    require(sha(HERE/'ENDPOINT_V2.md') == ENDPOINT_SHA, 'Additive matched-history contract')
    original = ORIGINAL.read_text()
    old_comparator = """    require(acquisitions['null']['null_full_output_bill_parity'],'Measured null parity')
    for null_call,control_call in ((0,0),(1,40)):
        a,b = acquisitions['null']['trials'][null_call],control['trials'][control_call]
        require(blobs['null'][null_call] == controls[control_call] and a['flops'] == b['flops']
                and a['operation_count'] == b['operation_count'], 'Null all16 BYTE/full bill exact')
"""
    new_comparator = """    require(acquisitions['null']['null_full_output_bill_parity'],'Measured matched-history null parity')
    require(acquisitions['null']['prerequisites']['bank_result_sha256'] ==
            pin(HERE/'native2_bank_v1/result.json'), 'Null used this completed fresh two-call bank')
    for fresh_call,control_call in ((0,0),(1,40)):
        bank,old = acquisitions['bank']['trials'][fresh_call],control['trials'][control_call]
        require(blobs['bank'][fresh_call] == controls[control_call] and bank['flops'] == old['flops'],
                'Direct-bank bytes and full bill equal archived corresponding cold/warm output')
        if fresh_call == 0:
            require(bank['operation_count'] == old['operation_count'], 'Same fresh cold-bank opcount')
        null = acquisitions['null']['trials'][fresh_call]
        require(blobs['null'][fresh_call] == blobs['bank'][fresh_call]
                and null['flops'] == bank['flops'] and null['operation_count'] == bank['operation_count'],
                'Same-history null all16 BYTE/full F/opcount exact on each call')
    result['matched_history_null_control'] = dict(passed=True,ranks=[37,37],fresh_processes=True,
        bank_result_sha256=pin(HERE/'native2_bank_v1/result.json'),
        null_result_sha256=pin(HERE/'native2_null_v1/result.json'),
        bank_immediate_warm_operation_count=acquisitions['bank']['trials'][1]['operation_count'],
        archived_after40_operation_count=control['trials'][40]['operation_count'],
        archived_warm_operation_counts_compared=False,
        score_benchmark='Original archived d50 VALID40 and its own original complete bills')
"""
    edits = [
        ("NATIVE_WATCH_SHA = '234db460741f6c42033e059a72f3a0e2c3100d3b240641fd38926bdcbcde0f2c'",
         'NATIVE_WATCH_SHA = '+repr(WATCH_V2_SHA), 1),
        ("HERE/'native.py'", "HERE/'native_v2.py'", 5),
        ("HERE/'native_watch.py'", "HERE/'native_watch_v2.py'", 2),
        ("    pin(HERE/'native_watch_v2.py',NATIVE_WATCH_SHA)\n",
         "    pin(HERE/'native_watch_v2.py',NATIVE_WATCH_SHA)\n"
         "    pin(HERE/'judge.py',ORIGINAL_JUDGE_SHA)\n"
         "    pin(HERE/'native.py',ORIGINAL_NATIVE_SHA)\n"
         "    pin(HERE/'ENDPOINT_V2.md',ENDPOINT_PROTOCOL_SHA)\n", 1),
        ("    for role,watch_stage,count in (('null','null',2),('terminal','native',41)):",
         "    for role,watch_stage,count in (('bank','bank',2),('null','null',2),('terminal','native',41)):", 1),
        ("HERE/f'native_{role}_v1'", "HERE/f'native2_{role}_v1'", 1),
        ("HERE/f'native_{role}_watch_v1'", "HERE/f'native2_{role}_watch_v1'", 1),
        ("        candidate = context['null' if role == 'null' else 'candidate']",
         "        candidate = tl.BANK if role == 'bank' else context['null' if role == 'null' else 'candidate']", 1),
        ("        candidate_sha = tl.BANK_SHA if role == 'null' else context['candidate_sha256']",
         "        candidate_sha = tl.BANK_SHA if role != 'terminal' else context['candidate_sha256']", 1),
        ("(n.VALID[0],) if role == 'null' else n.VALID",
         "(n.VALID[0],) if role != 'terminal' else n.VALID", 1),
        ("'current_d50_native_objective_p1_native_watch_v1'",
         "'current_d50_native_objective_p1_native_watch_v2'", 1),
        (old_comparator, new_comparator, 1),
        ("out = HERE/'score_v1'", "out = HERE/'score2_v1'", 1),
        ("schema='current_d50_native_objective_p1_score_v1'",
         "schema='current_d50_native_objective_p1_score_v2'", 1),
        ("        training_or_refit_performed=False,independent_panel_qualified=False,production_qualified=False,",
         "        original_judge_sha256=ORIGINAL_JUDGE_SHA,source_adapter_inverse_verified=True,\n"
         "        endpoint_protocol_sha256=ENDPOINT_PROTOCOL_SHA,\n"
         "        training_or_refit_performed=False,independent_panel_qualified=False,production_qualified=False,", 1),
    ]
    changed = original
    for before, after, count in edits:
        require(changed.count(before) == count, 'Exact endpoint2 scorer splice multiplicity')
        changed = changed.replace(before, after)
    inverse = changed
    for before, after, count in reversed(edits):
        require(inverse.count(after) == count, 'Exact endpoint2 scorer inverse multiplicity')
        inverse = inverse.replace(after, before)
    require(inverse == original, 'Full source inverse; all remaining v1 scorer statements unchanged')
    return changed


def main():
    if '--run' not in sys.argv:
        print('Inert endpoint2 scorer: same CLI as judge.py; reviewed source and completed endpoint2 receipts required.')
        return 0
    require(re.fullmatch('[0-9a-f]{64}', WATCH_V2_SHA) is not None,
            'V2 watcher has not been frozen; no scoring or imports admitted')
    source = derived_source()
    environment = dict(__file__=str(Path(__file__).resolve()),__name__='p1_preserved_endpoint2_judge',
        ORIGINAL_JUDGE_SHA=ORIGINAL_SHA,ORIGINAL_NATIVE_SHA=NATIVE_V1_SHA,ENDPOINT_PROTOCOL_SHA=ENDPOINT_SHA)
    exec(compile(source,str(__file__)+':preserved_score','exec'),environment)
    return environment['main']()


if __name__ == '__main__':
    raise SystemExit(main())
