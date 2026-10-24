"""Authenticate the existing public closed-context cleanup helper before import."""
import hashlib
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent.parent/'astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
spec = importlib.util.spec_from_file_location('qualified_closed_context_helper',SOURCE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
reset_saved_history = helper.reset_saved_history


def qualify(records,check,read):
    for r in records:
        for key in ['prediction','bill','weights','input_bill','input_reset','lifecycle']:
            check(r[key])
        ingress = read(r['input_bill']['path'])
        assert ingress['flops'] == 0 and ingress['context_closed'] and not ingress['charged_to_participant']
        for reset_key,bill_key in [('input_reset','input_bill'),('lifecycle','bill')]:
            x = read(r[reset_key]['path'])
            assert x['complete'] and x['context_released'] and x['closed_context_bill_unchanged_by_reset']
            assert x['after_context_exit'] and not x['candidate_state_reset']
            assert x['saved_bill'] == r[bill_key]['path'] and x['saved_bill_sha256'] == r[bill_key]['sha256']
            assert x['index'] == r['index'] and x['rank'] == r['rank']
