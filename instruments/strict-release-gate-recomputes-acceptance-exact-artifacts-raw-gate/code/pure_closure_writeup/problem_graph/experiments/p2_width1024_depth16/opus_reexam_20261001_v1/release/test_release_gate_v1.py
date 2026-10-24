"""Regression checks for the release-chain gate hole; no heavy sessions."""
import unittest
from release_gate_v1 import C_GATES, gate_errors, judge_errors


class ReleaseGateTests(unittest.TestCase):
    def test_exact_missing_worker_exception(self):
        body = '[designation x] INCOMPLETE: no result for: worker\n'
        self.assertEqual(judge_errors(body, 'x', True), [])
        self.assertTrue(judge_errors(body, 'x', False))

    def test_spoiled_missing_worker_is_not_fireable(self):
        body = '[designation x] INCOMPLETE: no result for: worker; also noted: pair paged (kernel counts unreliable)\n'
        self.assertTrue(judge_errors(body, 'x', True))

    def test_incomplete_pair_and_numeric_failure_are_not_fireable(self):
        self.assertTrue(judge_errors('[designation x] INCOMPLETE: no result for: pair, worker\n', 'x', True))
        self.assertTrue(judge_errors('  FAIL setup wall\n[designation x] INCOMPLETE: no result for: worker\n', 'x', True))

    def test_raw_failures_and_sha_mismatch_are_not_hidden(self):
        d = {'candidate_sha256': 'same', 'passed': True, 'gates': dict.fromkeys(C_GATES, True)}
        self.assertEqual(gate_errors('C', d, C_GATES, 'same'), [])
        for gate in C_GATES:
            bad = dict(d, gates=dict(d['gates'], **{gate: False}))
            self.assertTrue(gate_errors('C', bad, C_GATES, 'same'), gate)
        self.assertTrue(gate_errors('C', d, C_GATES, 'different'))
        self.assertTrue(gate_errors('C', dict(d, gates={}), C_GATES, 'same'))

    def test_wrong_tag_and_multiple_verdicts_rejected(self):
        self.assertTrue(judge_errors('[designation y] PASS: every criterion met\n', 'x', True))
        self.assertTrue(judge_errors('[designation x] PASS: every criterion met\n' * 2, 'x', True))


if __name__ == '__main__':
    unittest.main()
