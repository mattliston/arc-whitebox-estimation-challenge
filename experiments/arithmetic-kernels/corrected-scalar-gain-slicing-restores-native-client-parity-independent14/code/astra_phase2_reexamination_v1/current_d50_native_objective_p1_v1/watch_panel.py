"""P1 independent14 resource watch; inert, reversible TL2-watch adapter.

Only admission pins/paths, the changed-P acquisition predicates, and the
completed-bill PLUS completed-lifecycle progress boundary differ. The
inherited process-group supervision, limits and independent score decision
handling remain unchanged. No numerical library or worker is imported.
"""
import ast
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent/'current_d50_terminal_linear_v2/watch_panel.py'
ORIGINAL_SHA = '82919d983fbca86ffd418393f027a95d662ef1f18518613c831c1febccdeaaca'
PROTOCOL_SHA = '3985ac8107375b88c5452b455033ea2f13c3d166a3a5f3ed5de645962e1e5079'


LIFECYCLE_SOURCE = '''\
def validate_lifecycle(bill,bill_path,lifecycle_path):
    bill_path,lifecycle_path = Path(bill_path),Path(lifecycle_path)
    require(not bill_path.is_symlink() and not lifecycle_path.is_symlink() and
            bill_path.is_file() and lifecycle_path.is_file(),
            'Regular completed bill and lifecycle files')
    life = read(lifecycle_path)
    require(bill['lifecycle_receipt'] == str(lifecycle_path) and
            life['saved_bill'] == str(bill_path) and
            life['saved_bill_sha256'] == sha(bill_path) and
            life['call'] == bill['call'] and life['rank'] == bill['rank'] and
            life['reset_api'] == 'flopscope.budget_reset' and
            all(life.get(k) is True for k in ('complete','after_context_exit',
                'context_released','closed_context_bill_unchanged_by_reset')) and
            life['candidate_state_reset'] is False,
            'Completed public post-context reset binds the exact closed bill')
    pairs = (('flops','flops'),('operation_count','operation_count'),
             ('context_wall_s','context_wall_s'),('backend_s','flopscope_backend_s'),
             ('overhead_s','flopscope_overhead_s'),('residual_s','native_residual_s'))
    for closed_key,bill_key in pairs:
        value = life['closed_metrics'][closed_key]
        require(value == bill[bill_key] and math.isfinite(value) and value >= 0,
                'Closed bill metrics unchanged through release: '+closed_key)
    return life


'''


OLD_PROGRESS = '''\
            try:
                bill = read(path)
            except json.JSONDecodeError:
                break  # The one writer has not closed this newly created JSON yet.
            require(bill['call'] == call and bill['rank'] == CALLS[call] and
                    bill['includes_initialization'] is (call == 0) and
                    type(bill['flops']) is int and 0 < bill['flops'] <= BUDGET,
                    'Progress is only a complete next fixed prediction bill')
'''
NEW_PROGRESS = '''\
            lifecycle_path = child_out/f'call_{call:02d}_lifecycle.json'
            if not lifecycle_path.exists():
                break  # A saved bill alone does not retire its context.
            try:
                bill = read(path)
                validate_lifecycle(bill,path,lifecycle_path)
            except json.JSONDecodeError:
                break  # The sole writer has not closed both JSON receipts yet.
            require(bill['call'] == call and bill['rank'] == CALLS[call] and
                    bill['includes_initialization'] is (call == 0) and
                    type(bill['flops']) is int and 0 < bill['flops'] <= BUDGET,
                    'Progress requires the next complete fixed bill and released lifecycle')
'''


EDITS = (
    ("PROTOCOL_SHA = '7e2c4d86de5d600b19134fd3127d0878d05a527cab811ec36b2ebca4b3c69f2f'",
     "PROTOCOL_SHA = '"+PROTOCOL_SHA+"'",1),
    ("OLD_PROTOCOL_SHA = 'febed4bdccc7c031cb846e6d97f695536ca98d3bef10d07de48f1418a91b6a58'",
     "OLD_PROTOCOL_SHA = '2141e2d4f025522a7cbea55fbd5ff10a66c313feacb1b4bd49a13f75a4b9af35'",1),
    ("TERMINAL_SCORE_SHA = '9e327c7e7f8df5471679eb7532a4531b5a763ed6b051d32408e0231b6ceda030'",
     "TERMINAL_SCORE_SHA = '4f60a8def29f1a9326d68d5c25bbe79b3f26289af8cf46de81609fb91cf055e5'",1),
    ("TERMINAL_END_SHA = '61765bd814261708ca6a32c2bc8dee4828d465e9bf80fa558a186c59b61d4e63'",
     "TERMINAL_END_SHA = 'c72831702f18a70a4bc40a8d88a655879be3e816e3edd3351cce9ea75e8cd587'",1),
    ("TERMINAL_WATCH_SHA = '06a97b4215360af3446b1b166c78403ebf87d66ca0019ad6cd8eacaad6683e61'",
     "TERMINAL_WATCH_SHA = 'c9d015f7ed6b1da7db58bcf237582874042dd2d1e3315e72e7c55b5f1b9afa34'",1),
    ("TERMINAL_WATCH_END_SHA = 'fc36b81555a7b2dad7cdebd4e782bf7c7b949537b5b7ec1205e2de0fb462859e'",
     "TERMINAL_WATCH_END_SHA = 'fa220095fa711c7e80a9268aeb2e0ef410fb94444e775813e758d52eaa01d4f5'",1),
    ("CANDIDATE_SHA = '03d78ee8182749e6c5bed07618420bd9b134b242236a079dddba14837c823641'",
     "CANDIDATE_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'",1),
    ("SCHEMA = 'current_d50_terminal_linear_independent14_v1'",
     "SCHEMA = 'current_d50_native_objective_p1_independent14_v1'",1),
    ("WATCH_SCHEMA = 'current_d50_terminal_linear_independent14_watch_v1'",
     "WATCH_SCHEMA = 'current_d50_native_objective_p1_independent14_watch_v1'",1),
    ("P2 = HERE.parent.parent\n",
     "P2 = HERE.parent.parent\n"
     "HELPER_PATH = HERE.parent/'current_d50_terminal_linear_v2/watch_fit.py'\n"
     "ORIGINAL_PATH = HERE.parent/'current_d50_terminal_linear_v2/watch_panel.py'\n"
     "ORIGINAL_SHA = '"+ORIGINAL_SHA+"'\n",1),
    ("str(HERE/'watch_fit.py'):HELPER_SHA", "str(HELPER_PATH):HELPER_SHA",1),
    ("'tl_panel_resource_helpers',HERE/'watch_fit.py'",
     "'p1_panel_resource_helpers',HELPER_PATH",1),
    ("fit_v1/estimator.py", "build_v1/estimator.py",1),
    ("terminal_score_v1/", "score2_v1/",4),
    ("terminal_score_watch_v1/", "score2_watch_v1/",4),
    ("sources = {str(Path(__file__).resolve()):args.source_sha256,str(worker):args.worker_sha256,",
     "sources = {str(ORIGINAL_PATH):ORIGINAL_SHA,\n"
     "        str(Path(__file__).resolve()):args.source_sha256,str(worker):args.worker_sha256,",1),
    ("('passed','completed','hashes_unchanged','sources_unchanged','earlier15_rows_byte_exact',\n"
     "         'own_complete_F_and_opcounts_exact','all_outputs_and_bills_sealed_before_scoring',",
     "('passed','completed','hashes_unchanged','sources_unchanged','all16_domains_qualified',\n"
     "         'all47_selectors_qualified','own_complete_bills','all_outputs_and_bills_sealed_before_scoring',",1),
    ("    require(record['candidate_sha256'] == CANDIDATE_SHA and\n",
     "    require(record['mode'] == 'acquire','Only the fixed target-free acquisition mode')\n"
     "    require(record['candidate_sha256'] == CANDIDATE_SHA and\n",1),
    ("def validate_native(record,end,digest,args):\n",
     LIFECYCLE_SOURCE+"def validate_native(record,end,digest,args):\n",1),
    ("    first,last = record['trials'][0],record['trials'][-1]\n",
     "        bill_path = ROOT/'terminal'/f'call_{call:02d}_bill.json'\n"
     "        lifecycle_path = ROOT/'terminal'/f'call_{call:02d}_lifecycle.json'\n"
     "        require(read(bill_path) == trial and\n"
     "                record['artifacts'][str(bill_path)] == sha(bill_path) and\n"
     "                record['artifacts'][str(lifecycle_path)] == sha(lifecycle_path),\n"
     "                'Every declared trial binds its own complete bill and lifecycle')\n"
     "        validate_lifecycle(trial,bill_path,lifecycle_path)\n"
     "    first,last = record['trials'][0],record['trials'][-1]\n",1),
    ("result = dict(schema=WATCH_SCHEMA,stage=args.stage,passed=False,completed=False,hashes_unchanged=False,",
     "result = dict(schema=WATCH_SCHEMA,stage=args.stage,passed=False,completed=False,hashes_unchanged=False,\n"
     "        source_inverse_verified=True,original_watcher_sha256=ORIGINAL_SHA,\n"
     "        progress_requires_closed_lifecycle=True,",1),
    ("'The native progress bound includes initial admission and all between-call bookkeeping.',",
     "'The native progress bound includes initial admission and all between-call bookkeeping.',\n"
     "            'Only a matching complete prediction bill AND released-context lifecycle reset progress.',",1),
    (OLD_PROGRESS,NEW_PROGRESS,1),
    ("result['progress'].append(dict(call=call,rank=CALLS[call],path=str(path),observed_elapsed_s=elapsed))",
     "result['progress'].append(dict(call=call,rank=CALLS[call],path=str(path),\n"
     "                lifecycle_path=str(lifecycle_path),lifecycle_sha256=sha(lifecycle_path),\n"
     "                observed_elapsed_s=elapsed))",1),
    ("                item['sha256'] = sha(item['path'])\n",
     "                item['sha256'] = sha(item['path'])\n"
     "                require(sha(item['lifecycle_path']) == item['lifecycle_sha256'],\n"
     "                        'Observed completed lifecycle remains byte-exact')\n"
     "                validate_lifecycle(trial,item['path'],item['lifecycle_path'])\n",1),
)


def derived_source():
    """Stdlib-only source construction; no stage, subprocess or arrays."""
    original = ORIGINAL.read_bytes()
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA:
        raise RuntimeError('Frozen TL2 panel watchdog source changed')
    source = original.decode()
    for old,new,count in EDITS:
        if source.count(old) != count:
            raise RuntimeError('Watch adapter source-span mismatch: '+repr(old[:100]))
        source = source.replace(old,new)
    restored = source
    for old,new,count in reversed(EDITS):
        if restored.count(new) != count:
            raise RuntimeError('Watch adapter inverse-span mismatch: '+repr(new[:100]))
        restored = restored.replace(new,old)
    if restored.encode() != original:
        raise RuntimeError('Watch adapter inverse is not the frozen original bytes')
    ast.parse(source)
    return source


def namespace():
    """Load only the derived stdlib supervisor; its main remains uninvoked."""
    scope = dict(__file__=str(Path(__file__).resolve()),
                 __name__='p1_independent14_resource_watch',__doc__=__doc__)
    exec(compile(derived_source(),str(Path(__file__).resolve()),'exec'),scope)
    return scope


def main():
    return namespace()['main']()


if __name__ == '__main__':
    raise SystemExit(main())
