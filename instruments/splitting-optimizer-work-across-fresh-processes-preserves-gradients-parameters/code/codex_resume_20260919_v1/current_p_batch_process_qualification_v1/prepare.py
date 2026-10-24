"""Seal a bounded process-boundary study before any numerical worker starts."""
import ast
from datetime import datetime, timezone
from iso_common import *

assert not (HERE/'sources_sealed.json').exists()
assert not (HERE/'run_claim.json').exists() and not numerical_workers()
resource_floor()
assert pin(FIT/'sources_sealed.json')['sha256']=='6246722d91a2ec83930abd4f9753f077fd8fa4067d6534d107dd771f4b5e1747'
assert pin(FIT/'resource_failure.json')['sha256']=='4eb7dbe08d1209c05214d4ff92ecaceb405467974b673d7cb3c2e4ad1e3be400'
failure=read(FIT/'resource_failure.json')
assert failure['closed'] and not failure['passed'] and failure['no_terminal_payload']
assert failure['whole_group_retired'] and failure['no_restart_or_intermediate_salvage']
assert not (FIT/'terminal_payload.npz').exists()
old_seal=read(FIT/'sources_sealed.json')
for entry in old_seal['pins']:check(entry)
assert read(Q/'result.json')['passed'] and read(Q/'native_qualification.json')['passed']
assert read(Q/'torch_qualification.json')['passed']
watch=read(FIT/'fit_watch.json')
assert watch['exit_code']==1 and watch['whole_group_retired'] and not watch['remaining_rows']
assert watch['max_swap_growth_bytes']==0 and watch['max_rss_bytes']<7_500_000_000
inputs=read(FIT/'inputs.json')
order=read(FIT/'training_order.json')
first=[116,74,96,120,161,177,207,154]
assert order['seed']==9409212 and order['ranks'][:8]==first
assert len(order['ranks'])==160 and set(order['ranks'])==set(inputs['train'])
assert not set(first)&set(inputs['valid'])
data_pins=[inputs[k][str(rank)] for rank in first for k in ('weights','targets')]
for entry in data_pins:check(entry)
old_artifacts=[]
for index,rank in enumerate(first):
    path=FIT/f'train_{index:03d}_result.json'
    record=read(path)
    assert record['index']==index and record['rank']==rank and record['card_retired']
    assert record['retirement_memory']['after']['rss_bytes']<2_000_000_000
    assert not record['validation_targets_accessed']
    check(record['prediction'])
    old_artifacts.extend([path,Path(record['prediction']['path'])])
old_artifacts.extend(FIT/f'update_{i:02d}.json' for i in (1,2))

original=(FIT/'train.py').read_text()
source=original
replacements=[]
def replace(old,new):
    global source
    assert source.count(old)==1,old[:100]
    source=source.replace(old,new,1)
    replacements.append((old,new))

replace('from fit_common import *', 'import argparse\nfrom iso_common import *\nimport state_io')
replace("assert not (HERE / 'fit_result.json').exists() and not (HERE / 'training_order.json').exists()", """parser = argparse.ArgumentParser()
parser.add_argument('arm', choices=['reference', 'split0', 'split1'])
parser.add_argument('start', type=int)
parser.add_argument('stop', type=int)
parser.add_argument('--resume', type=Path)
args = parser.parse_args()
assert (args.arm, args.start, args.stop) in [('reference', 0, 2), ('split0', 0, 1), ('split1', 1, 2)]
BASE = HERE
if args.arm == 'split1':
    assert args.resume is not None and args.resume.resolve() == BASE/'split0/checkpoint_01.json'
else:
    assert args.resume is None
HERE = BASE/args.arm
HERE.mkdir(exist_ok=False)""")
replace("inputs = read(HERE / 'inputs.json')", "inputs = read(FIT / 'inputs.json')")
replace("recorder = allocation.Recorder(allocation.Allocator(), HERE / 'allocation_events.jsonl')", """state_io.initial_rng(args.arm, torch, np)
resume_receipt = None
if args.resume is not None:
    resume_receipt = state_io.restore(args.resume, args.start, torch, np, p, optimizer, model.PKEYS)
checkpoints = []
recorder = allocation.Recorder(allocation.Allocator(), HERE / 'allocation_events.jsonl')""")
replace('for update in range(40):', 'for update in range(args.start, args.stop):')
replace('        del card, weights, target, target_np, prediction, output, error, final_mse, earlier_mse, loss', """        record['accumulated_gradient_hashes'] = {k: model.tensor_sha(p[k].grad) for k in model.PKEYS}
        del card, weights, target, target_np, prediction, output, error, final_mse, earlier_mse, loss""")
replace("    path = HERE / f'update_{update + 1:02d}.json'", """    update_record['clipped_gradient_hashes'] = {k: model.tensor_sha(p[k].grad) for k in model.PKEYS}
    checkpoint = state_io.save(HERE, update + 1, torch, np, p, optimizer, model.PKEYS)
    checkpoints.append(checkpoint)
    update_record['checkpoint'] = checkpoint
    path = HERE / f'update_{update + 1:02d}.json'""")
tail=original[original.index('assert len(records) == 160 and len(updates) == 40'):]
replace(tail, """assert len(records) == 4 * (args.stop - args.start) and len(updates) == args.stop - args.start
assert len(checkpoints) == len(updates)
frozen_check()
terminal_hashes = {k: model.tensor_sha(v) for k, v in p.items()}
optimizer.zero_grad(set_to_none=True)
optimizer.state.clear()
parameters.clear()
gc.collect()
recorder.point('worker_complete')
recorder.stream.close()
result = dict(completed=True, passed=True, arm=args.arm, start_update=args.start, stop_update=args.stop,
    training_graphs=len(records), updates=len(updates), records=records, update_receipts=updates,
    checkpoints=checkpoints, restored_checkpoint=resume_receipt,
    initial_parameter_hashes=initial, terminal_parameter_hashes=terminal_hashes,
    trainable_keys=list(model.PKEYS), fixed_parameters_unchanged=True, inactive_parameters_unchanged=True,
    validation_targets_accessed=False, training_targets_accessed=True, checkpoint_selection=False,
    native_derivative_certified=False, native_score_measured=False, candidate_created=False,
    training_order=pin(HERE/'training_order.json'), source_seal=pin(BASE/'sources_sealed.json'),
    wall_s=time.monotonic()-began, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
verify()
resource_floor()
write(HERE/'result.json', result)
print(dict(completed=True, result=pin(HERE/'result.json'), wall_s=result['wall_s'],
    peak_rss_bytes=result['peak_rss_bytes']), flush=True)
""")
reversed_source=source
for old,new in reversed(replacements):
    assert reversed_source.count(new)==1
    reversed_source=reversed_source.replace(new,old,1)
assert reversed_source==original
ast.parse(source)
with (HERE/'worker.py').open('x') as stream:stream.write(source)
with (HERE/'guard.py').open('xb') as stream:stream.write((FIT/'guard.py').read_bytes())
assert (HERE/'guard.py').read_bytes()==(FIT/'guard.py').read_bytes()
write(HERE/'admission.json',dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),
    failed_fit_closed=pin(FIT/'resource_failure.json'),failed_fit_sources=pin(FIT/'sources_sealed.json'),
    original_training_source=pin(FIT/'train.py'),worker=pin(HERE/'worker.py'),
    source_inverse_byte_exact=True,source_replacements=[dict(before=old,after=new) for old,new in replacements],
    original_guard=pin(FIT/'guard.py'),guard=pin(HERE/'guard.py'),guard_byte_exact=True,
    ranks=first,maximum_backward_graphs=16,maximum_optimizer_updates=4,
    target_arrays_decoded=False,numerical_workers_started=False,no_partial_payload_used=True,
    no_full_fit_admitted=True,retired_graph_rss_limit_bytes=2_000_000_000,group_rss_limit_bytes=7_500_000_000,
    added_swap_acceptance_limit_bytes=0,free_disk_bytes=shutil.disk_usage(HERE).free))
external=[Path(p['path']) for p in old_seal['pins']]
external += [FIT/name for name in ['sources_sealed.json','resource_failure.json','fit_watch.json',
    'controller_result.json','inputs.json','training_order.json']]
external += old_artifacts
local=sorted(p for p in HERE.iterdir() if p.is_file())
for p in local:
    if p.suffix=='.py':ast.parse(p.read_text(),str(p))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(set(local+external))],
    data_pins=data_pins,math_and_limits_unchanged=True,maximum_backward_graphs=16,
    maximum_optimizer_updates=4,no_validation_arrays_decoded=True,no_full_fit_or_external_launch=True))
verify(data=True)
print(dict(prepared=True,sources=pin(HERE/'sources_sealed.json'),source_pins=len(read(HERE/'sources_sealed.json')['pins']),
    data_pins=len(data_pins),maximum_backward_graphs=16,maximum_optimizer_updates=4),flush=True)
