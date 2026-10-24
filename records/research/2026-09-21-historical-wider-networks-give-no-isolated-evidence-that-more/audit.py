"""Read historical result/shape metadata without importing a numerical runtime."""
import ast
import collections
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import pickletools
import sys
import zipfile

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
A = P2/'astra_phase2_reexamination_v1'
G = P2/'public_g22_operator_features_v1'


def pin(path):
    path = path.resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1 << 20),b''):
            h.update(b)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def read(path):
    return json.loads(path.read_text())


def rebuild(storage,offset,shape,stride,requires_grad,hooks,*rest):
    assert isinstance(storage,dict) and isinstance(shape,tuple)
    assert all(isinstance(x,int) and 0 <= x < 1_000_000 for x in shape)
    assert math.prod(shape) <= 2_000_000
    return dict(shape=list(shape),stride=list(stride),offset=offset,storage=storage)


class MetadataOnly(pickle.Unpickler):
    def find_class(self,module,name):
        # No original torch global is imported or executed. These local
        # substitutes construct only small descriptors, never tensor storage.
        return {('collections','OrderedDict'):collections.OrderedDict,
            ('torch._utils','_rebuild_tensor_v2'):rebuild,
            ('torch','FloatStorage'):'FloatStorage'}[(module,name)]

    def persistent_load(self,pid):
        assert isinstance(pid,tuple) and len(pid) == 5
        assert pid[0] == 'storage' and pid[1] == 'FloatStorage'
        return dict(dtype=pid[1],key=pid[2],device=pid[3],numel=pid[4])


def shapes(path):
    with zipfile.ZipFile(path) as z:
        infos = [x for x in z.infolist() if x.filename.endswith('/data.pkl')]
        assert len(infos) == 1 and infos[0].file_size < 100_000
        payload = z.read(infos[0])
        forbidden = {'STACK_GLOBAL','EXT1','EXT2','EXT4','INST','OBJ','NEWOBJ','NEWOBJ_EX'}
        assert all(op.name not in forbidden for op,arg,pos in pickletools.genops(payload))
        metadata = MetadataOnly(io.BytesIO(payload)).load()
        return dict(shape_metadata_bytes=len(payload),storage_members_decoded=False,
            shapes={k:v['shape'] for k,v in metadata.items()})


expected = {
    'GNN_H_C64_E40':'d70576c449eaefa6ec9d0d8735934c41943ed16078e01bb7a29cdddf3131c8b1',
    'GNN_H_C128_E80':'4d8708cce1e4eba46ded5ed40be9c1929147178b3acbc17b630b1d34406bf63a',
    'GNN_HGAUSS':'1b393ec0afd48a7107c2c7abe4203c716b2c32fa3ac9b2a1e7ae41a58988d7e3',
    'GNN_HGAUSS_CM48':'55055b4ef6fda049563189cb4a0a5b04f8a8a5629c6c51c83bd0f0e846f9fa5e',
    'GNN_HGAUSS_CM128':'4ccc7d75d23c38323efefa64c15545788a6527dffe17dbbbd43c5652379d924f',
    'GNN_HGPSD_CM48':'7b0fcc73e888699bfe9a94c05a2ea460673a324c2dc7dde9cd501056abbf52f7',
    'ASTRA_BINARY_PHYSICALGATE_V1_C64_E40':'12f660c0faeaa9b98b89cf5d1a080d5455a8f9da2697f5f412e9a1dc4179ee42',
}
records = []
for tag,sha in expected.items():
    path = G/f'GNN_{tag}_RESULT.json'
    assert pin(path)['sha256'] == sha
    r = read(path)
    model = G/f'GNN_{tag}_MODEL.pt'
    metadata = shapes(model)
    s = metadata['shapes']
    fdim = r['features']; cm = s['msg.weight'][0]//3
    assert s['msg.weight'][0] == 3*cm and s['msg.weight'][1] == r['channels']
    assert s['enc.0.weight'][1] == fdim+6*cm
    assert s['enc.4.weight'][0] == r['channels']
    records.append(dict(tag=tag,result=pin(path),model=pin(model),rows=r['rows'],
        features=fdim,state_width=r['channels'],message_width=cm,
        hidden_width=s['enc.0.weight'][0],epochs=r['epochs'],reported_parameters=r['n_params'],
        train_final=r['train_final'],valid_final=r['valid_final'],host_valid_final=r['valid_host_layer_mse'][-1],
        n_train=r['n_train'],n_valid=r['n_valid'],model_metadata=metadata,
        paired_current_native_score=False))

audit_path = D/'current_p_message_source_audit_v1/audit.json'
assert pin(audit_path)['sha256'] == '37d43005a382649a4ee830e4012a7a4ad3d090c59cfb4835deb8c88677db81df'
current = read(audit_path)
bank = Path(current['source']['path'])
assert pin(bank) == current['source']
source = bank.read_text()
tree = ast.parse(source)
gnn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'gnn_step')
function = ast.get_source_segment(source,gnn)
assert 'cm = int(P["msg_w"].shape[0])//3' in function
assert 'g = x[:,5:6]' in function
assert current['literal_shapes']['msg_w'] == [48,64]
assert current['literal_shapes']['enc0_w'] == [128,126]

gate_paths = [A/n for n in ['binary_physicalgate_training_v1.json',
    'binary_physicalgate_closedloop_r35_v1.json','binary_physicalgate_closedloop_r100_v1.json',
    'train_binary_physicalgate_v1.py','screen_binary_physicalgate_closedloop.py',
    'radial_gnn_physicalgate.py','RESEARCH_RECEIPTS_2026_09_04_LATE.md']]
closed = [read(A/f'binary_physicalgate_closedloop_r{rank}_v1.json')['final_mse'] for rank in [35,100]]
training = read(A/'binary_physicalgate_training_v1.json')
trainer = G/'train_gnn_oracle.py'
assert pin(trainer)['sha256'] == training['trainer_sha256']
trainer_source = trainer.read_text()
before = '            g = xl[:, 5:6]'
after = '            g = x[l, :, 5:6]'
assert trainer_source.count(before) == 1
assert hashlib.sha256(trainer_source.replace(before,after).encode()).hexdigest() == training['modified_source_sha256']
memory = A/'nonlinear_memory_capacity_current_v1'
memory_expected = {
    'training/result.json':'bdbf47a842093cbe4194da2bddf6328aedb88d7923bdb302d7b5be6d0352e383',
    'predictions/result.json':'e0663bf0202c4bc1bb31e00cf6726539d4f916c7cc8bf7433539c8266d008cbd',
    'consumers/result.json':'6bfe09d6abe5b8d2493f699684b9a8b4c42e6399eb3369d7a04acd94d6de2fea',
}
for path,sha in memory_expected.items():
    assert pin(memory/path)['sha256'] == sha

by_tag = {r['tag']:r for r in records}
report = '''# Historical capacity and gate evidence

This is a source/receipt audit, not a new estimator experiment. Existing
result JSONs and the small shape descriptors in saved PyTorch archives were
read with a restricted standard-library metadata decoder. No PyTorch global,
tensor storage, truth array, candidate inference, or training was executed.

The current bank already has three learned W/W²/W³ message groups and three
W²/W⁴/W⁶ energy groups, with state64, message16, hidden128 and30 features.
Message width is derived from the payload shape. A width change is therefore
not ruled out by a hardcoded16, but dimensions alone establish no accuracy,
native compatibility, cost, or resource benefit.

Actual wider fits exist. The old51-feature full-host state64/message16/
hidden128 fit has saved-row valid MSE7.35158e-9 after40epochs; state128 with
the same message16/hidden128 has7.57955e-9 after80epochs despite lower train
error. This is not an isolated message-width comparison. On a different
Gaussian host, state128/message48/hidden256 gives9.49439e-6, and state128/
message128/hidden512 gives9.37415e-6, both60epochs. Those fits change hidden
width as well as messages. The Gaussian-PSD message48 fit gives2.47649e-7.
These saved-row results do not measure the current endogenous estimator or
its own complete FLOP-adjusted score. They cannot establish a universal
capacity limit or forecast a current-bank gain. Serialized shapes, rather
than filename tags or today's trainer defaults, establish the dimensions.

The physical activation-probability gate replacement was actually refitted
on the frozen200 binary-host rows for40epochs. The training receipt binds
the exact trainer and its one-line source change. Saved-row held-out MSE
5.93605e-9 was slightly better than the documented6.01057e-9 control, but
the matched closed-loop rank35/100 screen gave2.25119507e-8/2.15027866e-8,
worse than the documented2.21231e-8/2.11385e-8 control. This spelling is an
executed negative, not an untested idea. Those historical screens are not
new metered native scores and do not test retaining additional raw-gate
channels alongside the existing standardized-gate messages.

The nonlinear-memory capacity directory also contains executed fits and
consumer receipts. It tested fixed16→8→8→2 plane decoders on an older teacher,
with six training networks and held-out400/500. Memory/control plane ratios
1.01105/1.01093 and mean-consumer ratios1.03505/1.03475 failed its original
gate. These are local teacher-forced plane/consumer discrepancies, not a
wider GNN test or an endogenous adjusted-score bound.

No new width, gate, feature, seed, learning schedule, candidate, or performance
threshold is selected by this audit. The active fixed fit and native endpoint
remain unchanged; their eventual native result determines their next gate.
'''
with (HERE/'RESULT.md').open('x') as f:
    f.write(report)
result = dict(utc=datetime.now(timezone.utc).isoformat(),completed=True,scope='historical source and result metadata only',
    records=records,current_message_audit=pin(audit_path),current_bank=pin(bank),
    current_width=dict(state=64,message=16,hidden=128,features=30,payload_driven_message_width=True),
    old_full_host_state_width_valid_ratio=by_tag['GNN_H_C128_E80']['valid_final']/by_tag['GNN_H_C64_E40']['valid_final'],
    old_gaussian_joint_width_valid_ratio=by_tag['GNN_HGAUSS_CM128']['valid_final']/by_tag['GNN_HGAUSS_CM48']['valid_final'],
    physical_gate=dict(executed_fit=True,exact_trainer_change_recovered=True,closed_loop_final_mse=closed,
        historical_receipts=[pin(p) for p in gate_paths],trainer=pin(trainer),current_native_experiment=False),
    nonlinear_memory=dict(executed=True,kind='older teacher-forced plane decoder, not GNN width',
        pins=[pin(memory/p) for p in memory_expected]+[pin(memory/'VERDICT_2026_09_07_PDT.md')]),
    new_model_or_protocol_selected=False,estimator_calls=0,new_training=False,target_arrays_opened=False,
    tensor_storage_decoded=False,numerical_modules_imported=sorted({'numpy','torch','flopscope'} & set(sys.modules)),
    report=pin(HERE/'RESULT.md'),audit_source=pin(HERE/'audit.py'))
assert not result['numerical_modules_imported']
with (HERE/'audit.json').open('x') as f:
    json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(dict(completed=True,historical_fits=len(records),estimator_calls=0,audit=pin(HERE/'audit.json')),flush=True)
