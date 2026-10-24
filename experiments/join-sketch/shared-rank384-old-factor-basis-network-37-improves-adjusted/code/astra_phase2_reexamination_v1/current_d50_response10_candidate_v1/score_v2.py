"""Versioned saved-FP32 admission correction; original score failure frozen."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = '3bc2aeb4d66a74984c2801c281a52e11b75a52f36d9bff5c877cca3daccbf2b4'
FAILED_SHA = 'ce4364aafa13c6277bfea4f5f9abda2207f44fac17db26d2a128ec777b4ce897'
AMENDMENT_SHA = '9995cd8ae375ae0decef7615e1b08678941510f00f1bd3bcbb7436a649bf30ac'


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derived_source():
    require(sha(HERE/'score.py') == ORIGINAL_SHA and sha(HERE/'SCORE_DTYPE_V2.md') == AMENDMENT_SHA,
            'Reviewed frozen scorer and dtype-only amendment')
    require(sha(HERE/'score_v1/result.json') == FAILED_SHA,'Preserved failed scorer receipt')
    failed = json.loads((HERE/'score_v1/result.json').read_text())
    end = json.loads((HERE/'score_v1/completion.json').read_text())
    require(failed['passed'] is False and failed['completed'] is False and failed['truth_accessed'] is False
            and failed['truth_bindings'] == {} and end['passed'] is False and end['completed'] is False
            and end['result_sha256'] == FAILED_SHA,'Original failure remains failed and target-free')
    original = (HERE/'score.py').read_text()
    edits = [
        ('value.dtype == np.float64','value.dtype == np.float32'),
        ("out = HERE/'score_v1'","out = HERE/'score_v2'"),
        ("schema='current_d50_response10_candidate_score_v1'",
         "schema='current_d50_response10_candidate_score_v2'"),
        ('    pins = {}; bind(sources,pins)\n',
         "    sources.update({str(HERE/'score.py'):ORIGINAL_SCORER_SHA,\n"
         "        str(HERE/'SCORE_DTYPE_V2.md'):DTYPE_AMENDMENT_SHA})\n"
         "    pins = {}; bind(sources,pins)\n"
         "    bind({str(HERE/'score_v1/result.json'):FAILED_SCORER_SHA,\n"
         "          str(HERE/'score_v1/completion.json'):sha(HERE/'score_v1/completion.json')},pins)\n"),
        ('        estimator_executed=False,automatic_next_stage=False,truth_accessed=False,\n',
         '        estimator_executed=False,automatic_next_stage=False,truth_accessed=False,\n'
         '        original_scorer_sha256=ORIGINAL_SCORER_SHA,dtype_amendment_sha256=DTYPE_AMENDMENT_SHA,\n'
         '        failed_original_result_sha256=FAILED_SCORER_SHA,source_inverse_verified=True,\n')]
    changed = original
    for before,after in edits:
        require(changed.count(before) == 1,'Unique dtype/metadata splice')
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        require(inverse.count(after) == 1,'Unique inverse splice')
        inverse = inverse.replace(after,before)
    require(inverse == original,'All remaining score/target/cost/resource statements unchanged')
    return changed


def main():
    if '--run' not in sys.argv:
        print('Inert dtype-only scorer adapter; same reviewed CLI as score.py.')
        return 0
    source = derived_source()
    namespace = dict(__file__=str(Path(__file__).resolve()),__name__='response10_corrected_scorer',
        ORIGINAL_SCORER_SHA=ORIGINAL_SHA,FAILED_SCORER_SHA=FAILED_SHA,DTYPE_AMENDMENT_SHA=AMENDMENT_SHA)
    exec(compile(source,str(__file__)+':preserved_score','exec'),namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
