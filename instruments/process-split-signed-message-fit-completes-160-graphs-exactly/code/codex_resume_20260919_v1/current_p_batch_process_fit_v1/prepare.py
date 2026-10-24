"""Admit one fixed full fit only after the separate exact handoff study passes."""
import ast
from datetime import datetime,timezone
from iso_common import *

assert not (HERE/'sources_sealed.json').exists() and not (HERE/'run_claim.json').exists()
assert not numerical_workers()
resource_floor()
qualification_result=read(QUAL/'result.json')
assert qualification_result['completed'] and qualification_result['passed'] and qualification_result['error'] is None
check(qualification_result['qualification'])
qualification=read(QUAL/'qualification.json')
assert qualification['passed'] and qualification['backward_graphs']==16 and qualification['optimizer_updates']==4
assert qualification['old_first_eight_graphs_exact'] and qualification['old_first_two_updates_exact']
assert qualification['all_parameter_adam_group_and_rng_states_exact']
assert qualification['every_worker_group_retired'] and not qualification['full_fit_launched']
assert pin(QUAL/'sources_sealed.json')['sha256']=='0a531588743f15cf79946b23a3590967c48c64283d5d22df1f3a6751b03c49f3'
for entry in read(QUAL/'sources_sealed.json')['pins']:check(entry)
for arm in ['reference','split0','split1','qualify']:
    watch=read(QUAL/f'{arm}_watch.json')
    assert watch['completed'] and watch['exit_code']==0 and watch['error'] is None
    assert watch['whole_group_retired'] and not watch['remaining_rows']
    assert watch['max_swap_growth_bytes']==0 and watch['max_rss_bytes']<7_500_000_000
claim=read(QUAL/'run_claim.json')
rows=subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,command='],text=True,timeout=5)
assert not [line for line in rows.splitlines() if len(fields:=line.strip().split(None,2))==3
    and (int(fields[0])==claim['pid'] or int(fields[1])==claim['pgid'])]
failure=read(FIT/'resource_failure.json')
assert failure['closed'] and failure['no_terminal_payload'] and not (FIT/'terminal_payload.npz').exists()
inputs=read(FIT/'inputs.json')
order=read(FIT/'training_order.json')
assert order['seed']==9409212 and len(order['ranks'])==160 and set(order['ranks'])==set(inputs['train'])
assert len(inputs['train'])==160 and len(inputs['valid'])==40 and not set(inputs['train'])&set(inputs['valid'])
data_pins=[inputs[k][str(rank)] for rank in order['ranks'] for k in ('weights','targets')]
for entry in data_pins:check(entry)
old_artifacts=[]
for index in range(134):
    path=FIT/f'train_{index:03d}_result.json';record=read(path)
    assert record['index']==index and record['rank']==order['ranks'][index]
    assert record['card_retired'] and record['retirement_memory']['after']['rss_bytes']<2_000_000_000
    assert not record['validation_targets_accessed'];check(record['prediction'])
    old_artifacts.extend([path,Path(record['prediction']['path'])])
old_artifacts.extend(FIT/f'update_{i:02d}.json' for i in range(1,34))
for index in range(8):
    assert all(prefix_graph(read(QUAL/'reference'/f'train_{index:03d}_result.json')).values())
for number in (1,2):
    assert all(prefix_update(read(QUAL/'reference'/f'update_{number:02d}.json')).values())

original=(QUAL/'worker.py').read_text();source=original;replacements=[]
def replace(old,new):
    global source
    assert source.count(old)==1,old[:100]
    source=source.replace(old,new,1);replacements.append((old,new))
replace('verify(data=True)\nparser = argparse.ArgumentParser()',
    'verify()\nparser = argparse.ArgumentParser()')  # Each graph still verifies its own data hashes.
parser_start=original.index('parser = argparse.ArgumentParser()')
parser_stop=original.index('HERE.mkdir(exist_ok=False)')+len('HERE.mkdir(exist_ok=False)')
replace(original[parser_start:parser_stop],"""parser = argparse.ArgumentParser()
parser.add_argument('update', type=int)
parser.add_argument('--resume', type=Path)
args = parser.parse_args()
assert 0 <= args.update < 40
args.start, args.stop = args.update, args.update + 1
args.arm = f'batch_{args.update:02d}'
BASE = HERE
if args.start:
    assert args.resume is not None and args.resume.resolve() == BASE/f'batch_{args.start-1:02d}'/f'checkpoint_{args.start:02d}.json'
else:
    assert args.resume is None
HERE = BASE/args.arm
HERE.mkdir(exist_ok=False)""")
replace('state_io.initial_rng(args.arm, torch, np)',"state_io.initial_rng('split0' if args.start == 0 else 'continuation', torch, np)")
replace("        record['card_retired'] = True","        record['card_retired'] = True\n        record['prefix_checks'] = prefix_graph(record)")
replace("    update_record['checkpoint'] = checkpoint","    update_record['checkpoint'] = checkpoint\n    update_record['prefix_checks'] = prefix_update(update_record)")
inverse=source
for old,new in reversed(replacements):
    assert inverse.count(new)==1
    inverse=inverse.replace(new,old,1)
assert inverse==original
ast.parse(source)
with (HERE/'worker.py').open('x') as stream:stream.write(source)
for name in ['state_io.py','guard.py']:
    with (HERE/name).open('xb') as stream:stream.write((QUAL/name).read_bytes())
    assert (HERE/name).read_bytes()==(QUAL/name).read_bytes()
rng=read(QUAL/'initial_rng.json')['payload'];check(rng)
with (HERE/'initial_rng.npz').open('xb') as stream:stream.write(Path(rng['path']).read_bytes())
write(HERE/'initial_rng.json',dict(payload=pin(HERE/'initial_rng.npz'),source=rng,no_seed_selection=True,
    original_zero_model_start=True,no_fitted_payload_used=True))
with (HERE/'training_order.json').open('xb') as stream:stream.write((FIT/'training_order.json').read_bytes())
write(HERE/'admission.json',dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),
    qualification=pin(QUAL/'qualification.json'),qualification_controller=pin(QUAL/'result.json'),
    qualification_controller_retired=True,failed_fit_closed=pin(FIT/'resource_failure.json'),
    original_training_source=pin(FIT/'train.py'),qualified_worker=pin(QUAL/'worker.py'),worker=pin(HERE/'worker.py'),
    source_inverse_byte_exact=True,source_replacements=[dict(before=old,after=new) for old,new in replacements],
    state_transport_byte_exact=True,guard_byte_exact=True,first_eight_prefix_comparators_exercised=True,
    first_two_boundary_comparators_exercised=True,old_graph_prefix_count=134,old_update_prefix_count=33,
    maximum_backward_graphs=160,maximum_optimizer_updates=40,initial_model='original_zero_branch',
    initial_rng=pin(HERE/'initial_rng.json'),target_arrays_decoded=False,fit_started=False,
    no_partial_payload_used=True,no_native_or_external_launch=True,free_disk_bytes=shutil.disk_usage(HERE).free))
external=[Path(entry['path']) for entry in read(QUAL/'sources_sealed.json')['pins']]
external += [p for p in QUAL.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
external += old_artifacts
local=sorted(p for p in HERE.iterdir() if p.is_file())
for path in local:
    if path.suffix=='.py':ast.parse(path.read_text(),str(path))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(set(local+external))],data_pins=data_pins,
    model_math_and_limits_unchanged=True,maximum_backward_graphs=160,maximum_optimizer_updates=40,
    no_validation_arrays_decoded=True,no_native_or_external_launch=True))
verify()
print(dict(prepared=True,sources=pin(HERE/'sources_sealed.json'),source_pins=len(read(HERE/'sources_sealed.json')['pins']),
    training_data_pins=len(data_pins),maximum_backward_graphs=160,maximum_optimizer_updates=40),flush=True)
