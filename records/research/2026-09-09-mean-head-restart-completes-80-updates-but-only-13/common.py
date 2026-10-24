"""Inverse-preserved P2 admission plus failed-prefix metadata; no new numerics."""
import hashlib
from pathlib import Path

ORIGINAL = Path(__file__).resolve().parent.parent/'current_p1_native_objective_p2_v1'
ORIGINAL_COMMON_SHA = 'f443ec27b2d9d1547d2ba755431226d4ac1d7ecba73fdd0e74b18dfd8434e1f9'
ORIGINAL_WORKER_SHA = '41cdc5993e60510f86401971f558bc370400027d08ba1f8295831be37e7b21a8'
ORIGINAL_WATCH_SHA = 'fe227690050e88265c972f67f14732578f1bb55aef73f58489f118d67ee5640d'
ORIGINAL_PROTOCOL_SHA = 'b2ceb8ac16086e19a5630658df7fb43afaa17b47d761267a1d84ffadce8d9b16'
RESTART_PROTOCOL_SHA = '968910dc63ba1c090665322c945b4e7db3b2978b1a059baa10bab13ea2dc3af7'
MONITOR_SHA = 'd4b3af53b716593670d8e0c7f0fc6f2133b57f1088d9966c06635d3a23f39873'
TEST_SOURCE_SHA = '1438ca2f4b12b53628cfbb2be4f997eea8949bdf5a70a8c44c74738e83529fb9'
FAILED_WATCH_SHA = 'a101ffc18acd2568721472a890bdb9fabdd00a86bf0a52240f2780553e904124'
FAILED_COMPLETION_SHA = '412ad282149327ee272ca8b8a873b10fac528c57d22e9ef15c14058e122a2ed8'
RESTART_SCHEMA = 'current_p1_native_objective_p2_monitor_v2'
RESTART_WATCH_SCHEMA = 'current_p1_native_objective_p2_monitor_watch_v2'
RESTART_ROSTER_SCHEMA = 'current_p1_native_objective_p2_monitor_roster_v2'
_original_bytes = (ORIGINAL/'common.py').read_bytes()
if hashlib.sha256(_original_bytes).hexdigest() != ORIGINAL_COMMON_SHA:
    raise RuntimeError('Frozen P2 common owner changed')
_original_text = _original_bytes.decode()
_edits = [(ORIGINAL_PROTOCOL_SHA,RESTART_PROTOCOL_SHA),
          ('current_p1_native_objective_p2_v1',RESTART_SCHEMA),
          ('current_p1_native_objective_p2_watch_v1',RESTART_WATCH_SCHEMA)]
_replacement = _original_text
for _before,_after in _edits:
    if _replacement.count(_before) != 1: raise RuntimeError('Unique common version substitution')
    _replacement = _replacement.replace(_before,_after)
_inverse = _replacement
for _before,_after in reversed(_edits):
    if _inverse.count(_after) != 1: raise RuntimeError('Unique common version inverse')
    _inverse = _inverse.replace(_after,_before)
if _inverse.encode() != _original_bytes: raise RuntimeError('Common source inverse failed')
RESTART_COMMON_PROOF = dict(parent_source_sha256=ORIGINAL_COMMON_SHA,inverse_source_byte_exact=True,
    substitution_count=len(_edits),changed_scope='protocol/schema identity only; explicit metadata hooks below')
exec(compile(_replacement,str(Path(__file__).resolve())+':frozen_p2_common','exec'),globals())

PKEYS_FIXED = ('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b','head','head_b','gain')
LOSS_FIELDS = ('raw_final_mse','earlier_mean_mse','unweighted_surrogate_loss',
               'weighted_surrogate_loss','normalized_d50_cost_weight')
MOCK_CHECKS = ('concurrent_delayed_within_actual_1s','timeout_no_retry_and_owned_reap',
    'late_finalization_is_failure','nonzero_command_is_failure','single_snapshot_tree_and_foreign_inventory',
    'malformed_duplicate_process_table_rejected','owned_group_cleanup_and_exit_race','exited_leader_live_group_is_retired')


def restart_source_bindings():
    return {str(ORIGINAL/name):digest for name,digest in (
        ('common.py',ORIGINAL_COMMON_SHA),('worker.py',ORIGINAL_WORKER_SHA),
        ('watch.py',ORIGINAL_WATCH_SHA),('PROTOCOL.md',ORIGINAL_PROTOCOL_SHA))} | {
        str(HERE/'monitor.py'):MONITOR_SHA,str(HERE/'monitor_tests.py'):TEST_SOURCE_SHA}


def mock_admission(digest, pin):
    folder = HERE/'monitor_tests_v1'; rp,cp = folder/'result.json',folder/'completion.json'
    require(isinstance(digest,str) and len(digest) == 64 and set(digest) <= set('0123456789abcdef'), 'Explicit mock-result SHA')
    pin(rp,digest); pin(cp,None)
    value,end = read(rp),read(cp)
    require(value['schema'] == 'current_p1_native_objective_p2_monitor_tests_v2'
            and all(value.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged'))
            and end['passed'] and end['completed'] and end['result_sha256'] == digest
            and tuple(value['checks']) == MOCK_CHECKS and value['real_subprocesses_started'] == value['real_signals_sent'] == 0
            and all(value[k] is False for k in ('numerical_imports','model_accessed','arrays_accessed','targets_accessed'))
            and value['elapsed_s'] < 5 and value['peak_rss_bytes'] < 256_000_000, 'Complete target-free mock certificate')
    expected = {str(HERE/'monitor.py'):MONITOR_SHA,str(HERE/'monitor_tests.py'):TEST_SOURCE_SHA,
        str(HERE/'PROTOCOL.md'):RESTART_PROTOCOL_SHA,str(P1/'watch.py'):P1_WATCH_SHA}
    require(value['source_bindings'] == expected,'Exactly reviewed mock owners')
    for path,source_sha in expected.items(): pin(path,source_sha,True)
    return dict(result_sha256=digest,completion_sha256=sha(cp),checks=list(MOCK_CHECKS),
                real_processes_or_signals=False)


def signature(value):
    require(set(value['gradient_raw_sha256']) == set(PKEYS_FIXED)
            and all(isinstance(x,str) and len(x) == 64 and set(x) <= set('0123456789abcdef')
                    for x in value['gradient_raw_sha256'].values()), 'All11 exact accumulated-gradient hashes')
    require(value['prediction']['dtype'] == 'float32' and value['prediction']['shape'] == [16,1024]
            and all(type(value[k]) in (int,float) and math.isfinite(value[k]) for k in LOSS_FIELDS), 'Finite exact served/loss signature')
    require(all(value[k] is True for k in ('all16_domains_present','all47_selectors_present',
                'observed_checkpoint_replay_byte_exact','frozen_cstar')), 'Original per-call numerical checks')
    return dict(graph_index=value['graph_index'],rank=value['rank'],kind=value['kind'],update=value['update'],
        served_raw_sha256=value['prediction']['raw_sha256'],served_file_sha256=value['prediction']['sha256'],
        gradient_raw_sha256=value['gradient_raw_sha256'],forward_sha256=value['forward_diagnostics']['sha256'],
        replay_sha256=value['diagnostics']['sha256'],**{k:value[k] for k in LOSS_FIELDS})


def prefix_provenance(pin):
    """Only JSON/hash work; no target/parameter/served numerical array decode."""
    folder = ORIGINAL/'training_watch_v1'; rp,cp = folder/'result.json',folder/'completion.json'
    pin(rp,FAILED_WATCH_SHA); pin(cp,FAILED_COMPLETION_SHA)
    failed,end = read(rp),read(cp)
    require(failed['schema'] == 'current_p1_native_objective_p2_watch_v1' and failed['stage'] == 'train'
            and failed['passed'] is False and failed['completed'] is False and end['passed'] is False
            and end['completed'] is False and end['result_sha256'] == FAILED_WATCH_SHA
            and failed['whole_owned_tree_retired'] and failed['completed_graph_count'] == 199
            and len(failed['graph_progress']) == 199 and 'TimeoutExpired' in failed['error']
            and 'timed out after 0.4 seconds' in failed['error'] and "'/bin/ps'" in failed['error'], 'Preserved operational failure, never a PASS')
    for path,digest in restart_source_bindings().items(): pin(path,digest,True)
    for name,digest in (('common.py',ORIGINAL_COMMON_SHA),('worker.py',ORIGINAL_WORKER_SHA),
                        ('watch.py',ORIGINAL_WATCH_SHA),('PROTOCOL.md',ORIGINAL_PROTOCOL_SHA)):
        require(failed['source_bindings'][str(ORIGINAL/name)] == digest,'Failed run exact immutable source owner')
    # Authenticate the failed watch's complete non-target inventories, including
    # completed markers. Its target bindings are never traversed.
    for group in ('source_bindings','input_bindings','artifacts'):
        for path,digest in failed[group].items(): pin(path,digest,group == 'source_bindings')
    roster_path = ORIGINAL/'training_v1/model_call_roster.json'
    require(str(roster_path) in failed['artifacts'],'Old fixed roster sealed by failed watch')
    roster = read(roster_path)
    require(roster['schema'] == 'current_p1_native_objective_p2_roster_v1' and roster['stage'] == 'train'
            and roster['seed'] == SEED and roster['generator'] == 'PCG64DXSM'
            and roster['stream_permutations'] == [2,3] and len(roster['entries']) == 360,'Same original fixed restart trajectory')
    signatures = []
    for index,item in enumerate(failed['graph_progress']):
        path = ORIGINAL/'training_v1'/f'graph_{index:04d}_complete.json'
        require(item['graph_index'] == index and Path(item['path']) == path
                and failed['artifacts'][str(path)] == item['sha256'],'Exactly199 completed marker owners')
        pin(path,item['sha256']); value = read(path); entry = roster['entries'][index]
        require(all(value[k] is True for k in ('passed','completed','graph_retired'))
                and all(value[k] == entry[k] for k in ('graph_index','rank','kind','update'))
                and value['kind'] == 'train_backward' and value['update'] == index//4,'Completed fixed-prefix call')
        for key,suffix in (('prediction','served.npy'),('forward_diagnostics','forward.json'),('diagnostics','diagnostics.json')):
            info = value[key]; expected = ORIGINAL/'training_v1'/f'graph_{index:04d}_{suffix}'
            require(Path(info['path']) == expected,'Exact old linked output/domain artifact')
            pin(expected,info['sha256'])
        signatures.append(signature(value))
    return dict(failed_watch_sha256=FAILED_WATCH_SHA,failed_completion_sha256=FAILED_COMPLETION_SHA,
        old_failure_still_failed=True,completed_prefix_count=199,prefix_signatures=signatures,
        old_training_order=roster['train_order'],old_train_orders=roster['train_orders'],
        old_validation_order=roster['validation_order'],old_roster_sha256=sha(roster_path),
        targets_decoded=False,arrays_decoded=False,optimizer_or_payload_resumed=False)


# Only parser metadata is extended; the original Stage numerical guard and
# all model/data/optimizer functions are unchanged.
_argument_node = next(n for n in ast.parse(_derived).body if isinstance(n,ast.FunctionDef) and n.name == 'arguments')
_argument_source = ''.join(_derived.splitlines(keepends=True)[_argument_node.lineno-1:_argument_node.end_lineno])
_argument_old = '    args = parser.parse_args()\n'
_argument_new = "    parser.add_argument('--monitor-test-result-sha256')\n"+_argument_old
require(_argument_source.count(_argument_old) == 1,'Unique added mock-result CLI')
_argument_changed = _argument_source.replace(_argument_old,_argument_new)
require(_argument_changed.replace(_argument_new,_argument_old) == _argument_source,'Exact argument-parser inverse')
exec(compile(_argument_changed,str(__file__)+':mock_seal_argument','exec'),globals())

_original_admit, _original_complete, _original_finish = Stage.admit,Stage.graph_complete,Stage.finish


def _restart_admit(self):
    _original_admit(self)
    def pin(path,digest,source=False):
        return self.pin(path,digest,'source_bindings' if source else 'input_bindings')
    self.restart_prefix = prefix_provenance(pin)
    self.monitor_certificate = mock_admission(self.args.monitor_test_result_sha256,pin)
    self.record.update(restart_common_inverse=RESTART_COMMON_PROOF,
        monitor_certificate=self.monitor_certificate,failed_training_watch_sha256=FAILED_WATCH_SHA,
        old_operational_failure_still_failed=True,fresh_full_restart=True,partial_payload_or_optimizer_resumed=False,
        prefix199_authenticated_before_targets=True,prefix199_exact_matches=0,prefix199_required=199,
        restart_provenance={k:v for k,v in self.restart_prefix.items() if k != 'prefix_signatures'})
    require(self.record['training_targets_accessed'] is False,'Failure/signatures authenticated before every target access')
    self.json('restart_admission.json',dict(source_bindings=self.record['source_bindings'],
        input_bindings=self.record['input_bindings'],monitor_certificate=self.monitor_certificate,
        restart_provenance=self.record['restart_provenance'],training_targets_accessed=False,
        validation_targets_accessed=False))


def _restart_complete(self,index,item):
    if self.args.stage == 'train' and index < 199:
        require(signature(item) == self.restart_prefix['prefix_signatures'][index], 'Exact preserved199-call numerical signature')
        require(self.record['prefix199_exact_matches'] == index,'Monotone old-prefix comparison')
        self.record['prefix199_exact_matches'] += 1
    return _original_complete(self,index,item)


def _restart_finish(self):
    require(self.record['prefix199_exact_matches'] == (199 if self.args.stage == 'train' else 0),
            'Entire required prefix checked; qualification is not training')
    self.record['prefix199_byte_exact'] = self.args.stage == 'train'
    return _original_finish(self)


Stage.admit,Stage.graph_complete,Stage.finish = _restart_admit,_restart_complete,_restart_finish
