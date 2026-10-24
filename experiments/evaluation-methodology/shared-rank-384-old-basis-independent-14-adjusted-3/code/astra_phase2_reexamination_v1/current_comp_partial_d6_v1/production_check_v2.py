"""Inert source adapter for the fixed CD production checker.

The frozen v1 source is preserved. Only exact historical metadata schemas,
failure-only seal publication and the reviewed v2 source owners change.
No estimator or numerical library is imported by this module.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE / 'production_check.py'
OWNER_SHA = '0b55774c6878939920f65a4dfde32ba58755a07c42d57b1779ae32c777954670'
PROTOCOL_SHA = '5689f010691f45cafacd5fb97a894b3cac1b9fe6695807d3a7e03c7917682196'

OLD_TRUTHS = """    excluded_truths = {str(Path(v['path']).resolve()) for name in ('score40', 'panel_score')
                       for v in records[name]['truth_bindings'].values()}
"""
NEW_TRUTHS = """    # Two distinct, actual sealed schemas; neither opens a truth file.
    score_truths = records['score40']['truth_bindings']
    panel_truths = records['panel_score']['truth_bindings']
    require(isinstance(score_truths, dict) and score_truths and
            all(isinstance(p, str) and Path(p).is_absolute() and
                isinstance(s, str) and len(s) == 64 for p, s in score_truths.items()),
            'Native40 truth metadata is absolute path to SHA')
    require(isinstance(panel_truths, dict) and panel_truths and
            all(isinstance(rank, str) and isinstance(v, dict) and
                isinstance(v.get('path'), str) and Path(v['path']).is_absolute() and
                isinstance(v.get('sha256'), str) and len(v['sha256']) == 64
                for rank, v in panel_truths.items()), 'Panel truth metadata is rank to metadata')
    excluded_truths = {str(Path(p).resolve()) for p in score_truths}
    excluded_truths.update(str(Path(v['path']).resolve()) for v in panel_truths.values())
"""

FAILURE_SOURCE = r'''
def failure_seals(folder, result):
    """Bounded false receipts only; expired success guards cannot veto FAIL."""
    require(result['passed'] is False and result['completed'] is False, 'Failure-only sealing')
    signal.alarm(0)
    def publish(name, record):
        encoded = (json.dumps(record, indent=2, allow_nan=False) + '\n').encode()
        require(len(encoded) <= 65536, 'Bounded false seal, at most64KiB')
        path = folder / name
        with path.open('xb') as stream:
            stream.write(encoded)
        actual = path.read_bytes()
        require(actual == encoded, 'Actual exclusive false seal readback')
        return hashlib.sha256(actual).hexdigest()
    # Existing launch/pretruth files are retained; no large inventory or array is
    # read after failure. This receipt never claims a resource-qualified stage.
    failed = dict(schema=result['schema'], stage=result['stage'], passed=False,
        completed=False, production_qualified=False, candidate_sha256=SHA,
        failure_only=True, positive_admission_authority=False,
        estimator_executed=False, uploaded=False, truth_accessed=result['truth_accessed'],
        error=str(result.get('error', 'Checker failure'))[:8192],
        elapsed_s=time.monotonic() - STARTED,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    digest = publish('result.json', failed)
    completion = publish('completion.json', dict(passed=False, completed=False,
        result_sha256=digest))
    final = publish('finalization.json', dict(passed=False, completed=False,
        result_sha256=digest, completion_sha256=completion,
        elapsed_s=time.monotonic() - STARTED))
    print(json.dumps(dict(passed=False, stage=result['stage'], result_sha256=digest,
        completion_sha256=completion, finalization_sha256=final)), flush=True)
    return 1

'''


def derived_source():
    original = OWNER.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != OWNER_SHA:
        raise AssertionError('Frozen production checker v1 owner')
    edits = [
        ("'PRODUCTION_PROTOCOL.md'", "'PRODUCTION_PROTOCOL_V2.md'", 2),
        ("'production_watch.py'", "'production_watch_v2.py'", 2),
        ("'PRODUCTION_WATCH_SPEC.md'", "'PRODUCTION_WATCH_SPEC_V2.md'", 2),
        ('FROZEN = {\n', 'FROZEN = {\n'
            f"    HERE / 'production_check.py': '{OWNER_SHA}',\n"
            f"    HERE / 'PRODUCTION_PROTOCOL.md': '{PROTOCOL_SHA}',\n", 1),
        (OLD_TRUTHS, NEW_TRUTHS, 1),
        ('def main():\n', FAILURE_SOURCE + 'def main():\n', 1),
        ("    code, proof = source_helpers()\n", "    code, proof = source_helpers()\n"
            "    proof['production_checker_v2_adapter'] = V2_SOURCE_PROOF\n", 1),
        ("    digest = write(folder / 'result.json', result)\n",
         "    if not result['passed']:\n        return failure_seals(folder, result)\n"
         "    digest = write(folder / 'result.json', result)\n", 1),
    ]
    source = original
    for old, new, count in edits:
        if source.count(old) != count or new in source:
            raise AssertionError('Exact v2 source-adapter anchor count')
        source = source.replace(old, new)
    inverse = source
    for old, new, count in reversed(edits):
        if inverse.count(new) != count:
            raise AssertionError('Exact v2 inverse anchor count')
        inverse = inverse.replace(new, old)
    if inverse != original:
        raise AssertionError('Byte-exact frozen v1 source inverse')
    proof = dict(owner_path=str(OWNER), owner_sha256=OWNER_SHA,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),
        exact_v1_source_inverse=True, numerical_body_unchanged=True,
        success_guards_and_resource_limits_unchanged=True,
        modifications=['two explicit truth-metadata schemas', 'bounded false seals only',
                       'v2 source ownership and preserved v1 dependency pins'])
    return source, proof


def runtime():
    source, proof = derived_source()
    namespace = dict(__name__='cd_production_checker_v2_derived',
                     __file__=str(Path(__file__).resolve()), V2_SOURCE_PROOF=proof)
    exec(compile(source, str(OWNER) + ':v2-metadata-only', 'exec'), namespace)
    return namespace


def main():
    return runtime()['main']()


if __name__ == '__main__':
    raise SystemExit(main())
