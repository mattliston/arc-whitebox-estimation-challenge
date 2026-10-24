"""Separate saved-output score supervisor; no numerical work on import."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
SCORER_SHA = '01bef7f3533518fda8fd78e005935124e10079ee87802b2ee65aee79ee042bc0'
if hashlib.sha256((HERE/'panel_score.py').read_bytes()).hexdigest() != SCORER_SHA:
    raise RuntimeError('Exact reviewed saved-output scorer')
import panel_score as s
c = s.score_helper()
OWNER = c.COMP/'panel_watch.py'
OWNER_SHA = '324bcd2a43f931f1a38f0f472e99cf593c9a08d8e2f1f63c0ca5ff5ef0d5c904'


def source():
    c.require(c.sha(OWNER) == OWNER_SHA,'Unchanged independent score supervisor owner')
    original = c.function_source(OWNER,'score_watch')
    changed = c.splice(original,[
        ('def score_watch(args):','def owned_score_watch(args):'),
        ('    began = last = time.monotonic()\n','    began = WATCH_BEGAN; last = time.monotonic()\n'),
        ('schema=c.WATCH_SCHEMA,stage=\'score\'',
         "schema=c.PREFIX+'score_watch_v1',stage='score'"),
        ("        scientific_advancement=False,truth_arrays_opened=False,numerical_imports=False,\n",
         "        scientific_advancement=False,truth_arrays_opened=False,numerical_imports=False,\n"
         "        watcher_source_inverse=WATCH_PROOF,new_three_seal_score_required=True,\n"),
        ('        prior,_ = c.sealed(c.WATCH_OUT,args.acquisition_watch_sha256,inputs,tick=deadline)\n',
         '        prior,_,_ = c.seal3(c.WATCH_OUT,args.acquisition_watch_sha256,inputs,deadline)\n'
         "        c.require(args.context_sha256 is None or args.context_sha256 == prior['admission_context_sha256'],\n"
         "                  'Explicit context equals the native watch context')\n"
         "        c.require(args.admission_watch_sha256 is None or args.admission_watch_sha256 == prior['admission_watch_sha256'],\n"
         "                  'Explicit retired handoff equals the native watch handoff')\n"),
        ("            rp,cp = c.SCORE_OUT/'result.json',c.SCORE_OUT/'completion.json'\n",
         "            rp,cp,fp = (c.SCORE_OUT/name for name in ('result.json','completion.json','finalization.json'))\n"),
        ('            record,end = c.sealed(c.SCORE_OUT,digest,inputs,tick=deadline)\n',
         '            record,end,final = c.seal3(c.SCORE_OUT,digest,inputs,deadline,limit=120)\n'
         "            c.require(all(record.get(key) is True for key in (\n"
         "                'exact_COMP14_rows_and_three_summaries_replay','all15_new_outputs_verified',\n"
         "                'all90_artifacts_qualified','all15_domains_and_selectors_reverified',\n"
         "                'new_three_seal_acquisition_verified','old_COMP_two_seal_ownership_preserved',\n"
         "                'no_tuning_or_new_scientific_choice')) and record['full_model_calls'] == 0\n"
         "                and record['estimator_executed'] is False and record['MPS_executed'] is False,\n"
         "                'Full current panel, unchanged arithmetic and no new model call')\n"),
        ("            result.update(child_result_sha256=digest,child_completion_sha256=c.sha(cp,deadline),\n",
         "            result.update(child_result_sha256=digest,child_completion_sha256=c.sha(cp,deadline),\n"
         "                child_finalization_sha256=c.sha(fp,deadline),new_three_seal_score_verified=True,\n"
         "                scientific_advancement=record['retain_local_component'],\n"),
        ("                                 (c.SCORE_OUT,('result.json','completion.json'))):\n",
         "                                 (c.SCORE_OUT,('result.json','completion.json','finalization.json'))):\n"),
        ("                          c.sha(c.SCORE_OUT/'completion.json',deadline) == result['child_completion_sha256'],\n"
         "                          'Final score result/completion seals')\n",
         "                          c.sha(c.SCORE_OUT/'completion.json',deadline) == result['child_completion_sha256'] and\n"
         "                          c.sha(c.SCORE_OUT/'finalization.json',deadline) == result['child_finalization_sha256'],\n"
         "                          'Final score actual three seals')\n"
         "                result.update(c.final_bind((sources,inputs,result['artifacts']),deadline))\n"
         "                deadline()\n"),
        ('    except BaseException:\n        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())\n',
         '    except BaseException:\n        c.set_hash_guard(None)  # Failed seal publication only; cannot restore PASS.\n'
         '        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())\n'),
        ('        except BaseException:\n            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,\n',
         '        except BaseException:\n            c.set_hash_guard(None)  # Failed seal publication only.\n'
         '            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,\n'),
    ])
    return changed,dict(owner=str(OWNER),owner_sha256=OWNER_SHA,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),whole_source_inverse_verified=True,
        live_child_monitor_and_resource_limits_unchanged=True,
        only_current_score_metadata_three_seals_and_fresh_union_added=True,
        dead_child_metadata_checks_inside_whole_limit_not_live_sampling_claim=True)


def main():
    began = time.monotonic()
    args = s.score_arguments('watcher')
    if args is None:
        return 0
    c.require(args.scorer_sha256 == SCORER_SHA,'Explicit fixed scorer owner')
    def guard():
        c.require(time.monotonic()-began < 120,'Whole score supervisor including source construction and publication')
    c.set_hash_guard(guard)
    try:
        derived,proof = source()
        namespace = dict(globals(),WATCH_PROOF=proof,WATCH_BEGAN=began)
        exec(compile(derived,str(OWNER)+':current-CD-score-three-seals','exec'),namespace)
        code = namespace['owned_score_watch'](args)
        rp,cp,fp = (c.SCORE_WATCH_OUT/name for name in ('result.json','completion.json','finalization.json'))
        record,end = c.read(rp),c.read(cp)
        if record['passed']:
            c.require_fresh_union(record)
        rh,ch = c.sha(rp),c.sha(cp)
        elapsed = time.monotonic()-began
        passed = bool(code == 0 and record['passed'] and record['completed'] and
            record['whole_owned_tree_retired'] and record['new_three_seal_score_verified'] and
            end['passed'] and end['completed'] and end['result_sha256'] == rh and elapsed < 120)
        c.write(fp,dict(passed=passed,completed=passed,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
        print(json.dumps({key:record.get(key) for key in ('passed','completed','retain_local_component',
            'elapsed_s','maximum_sampled_rss_bytes','maximum_sample_interval_s','error','final_sealing_error')},
            allow_nan=False),flush=True)
        return 0 if passed and c.sha(rp) == rh and c.sha(cp) == ch and time.monotonic()-began < 120 else 1
    finally:
        c.set_hash_guard(None)


if __name__ == '__main__':
    raise SystemExit(main())
