from pathlib import Path
import ast, hashlib, json
H=Path(__file__).resolve().parent.parent
D=Path(__file__).resolve().parent
old=(H/'hill_climb_rrpi_extra32_warm_v1/train.py').read_text()
s=old.replace('One fixed eight-update RRPI extra32 warm proposal. Root guarded execution only.','Fixed RRPI full80 continuation in four-update, optimizer-preserving segments.')
s=s.replace("assert len(sys.argv) == 2 and pin(protocol_path)['sha256'] == sys.argv[1]","assert len(sys.argv) == 3 and pin(protocol_path)['sha256'] == sys.argv[1]\nSEGMENT = int(sys.argv[2])\nassert 0 <= SEGMENT < 20\nSTART, STOP = 4 * SEGMENT, 4 * (SEGMENT + 1)")
s=s.replace("assert protocol['updates'] == 8 and protocol['batch'] == 4 and protocol['schedule_steps'] == 120","assert protocol['updates'] == 80 and protocol['batch'] == 4 and protocol['schedule_steps'] == 120")
s=s.replace("for item in protocol['data']:\n    for key in ('weights', 'truth'):\n        assert pin(item[key]['path']) == item[key], item[key]['path']","assert len(protocol['data']) == 160")
s=s.replace("OUT = HERE / 'training'\nOUT.mkdir()","OUT = HERE / ('segment_%02d' % SEGMENT)\nOUT.mkdir()\nCKPT = HERE / ('resume_after_%03d.pt' % STOP)\nassert not CKPT.exists()")
s=s.replace("assert len(RANKS) == len(set(RANKS)) == 32\nassert not set(RANKS) & (set(parts['train']) | set(parts['valid']))\nassert not set(RANKS) & {35, 100, 36, 45, 52, 58, 64, 70}","assert len(RANKS) == len(set(RANKS)) == 160\nassert RANKS == parts['train'][:160]\nassert not set(RANKS) & set(parts['valid'])\nrng = np.random.default_rng(0)\nORDER = rng.permutation(RANKS).tolist() + rng.permutation(RANKS).tolist()\nBATCHES = [ORDER[i:i+4] for i in range(0, 320, 4)]\nDATA = {item['rank']:item for item in protocol['data']}\nfor rank in {r for b in BATCHES[START:STOP] for r in b}:\n    for key in ('weights', 'truth'):\n        assert pin(DATA[rank][key]['path']) == DATA[rank][key]")
s=s.replace("def restore_frozen():\n    with torch.no_grad():\n        for key in ('p.head', 'p.head_b'):\n            T[key][[0, 15]] = torch.from_numpy(np.array(arrays[key][[0, 15]], copy=True))\n        T['c'][:, [1, 3]] = torch.from_numpy(np.array(arrays['c'][:, [1, 3]], copy=True))\n","def restore_frozen():\n")
s=s.replace("updates=8, batch=4, schedule_steps=120,","updates=80, batch=4, schedule_steps=120, segment=SEGMENT, start=START, stop=STOP,")
anchor="opt = torch.optim.Adam([dict(params=g['params'], lr=g['base']) for g in groups])\n"
assert s.count(anchor)==1
s=s.replace(anchor,anchor+"""
resume_pin = None
if START:
    previous = HERE / ('resume_after_%03d.pt' % START)
    prior = json.loads((HERE / ('segment_%02d' % (SEGMENT-1)) / 'completed.json').read_text())
    assert pin(previous) == prior['checkpoint']
    resume_pin = pin(previous)
    sv = torch.load(previous, map_location='cpu', weights_only=True)
    assert sv['update'] == START and sv['protocol_sha'] == sys.argv[1]
    assert sv['order'] == ORDER and set(sv['parameters']) == set(T)
    with torch.no_grad():
        for key in T:
            T[key].copy_(sv['parameters'][key])
        for key in DL:
            DL[key].copy_(sv['dials'][key])
    opt.load_state_dict(sv['optimizer'])
    torch.set_rng_state(sv['torch_rng'])
    # Reload exact parameter and optimizer tensors; no fresh Adam or scheduler reset.
    for key in T:
        assert torch.equal(T[key].detach(), sv['parameters'][key])
    assert all(float(DL[key].detach()) == float(sv['dials'][key]) for key in DL)
    del sv
restore_frozen()
""")
s=s.replace("for update in range(8):","for update in range(START, STOP):")
s=s.replace("for item in protocol['data'][4 * update:4 * update + 4]:\n            record = network_backward(item)","for rank in BATCHES[update]:\n            record = network_backward(DATA[rank])")
s=s.replace("        p['head'].grad[[0, 15]] = 0\n        p['head_b'].grad[[0, 15]] = 0\n        T['c'].grad[:, [1, 3]] = 0\n","")
cut="restore_frozen()\nendpoint = {key: value.detach().cpu().numpy().copy() for key, value in T.items()}"
assert s.count(cut)==1
s=s[:s.index(cut)]+"""restore_frozen()
checkpoint = dict(update=STOP, protocol_sha=sys.argv[1], order=ORDER,
                  parameters={key:value.detach().cpu().clone() for key,value in T.items()},
                  dials={key:value.detach().cpu().clone() for key,value in DL.items()},
                  optimizer=opt.state_dict(), torch_rng=torch.get_rng_state())
with CKPT.open('xb') as stream:
    torch.save(checkpoint, stream)
endpoint_pin = dials_pin = None
if STOP == 80:
    endpoint = {key: value.detach().cpu().numpy().copy() for key, value in T.items()}
    with (HERE / 'endpoint.npz').open('xb') as stream:
        np.savez(stream, **endpoint)
    write(HERE / 'endpoint_dials.json', {key: float(value.detach()) for key, value in DL.items()})
    endpoint_pin, dials_pin = pin(HERE / 'endpoint.npz'), pin(HERE / 'endpoint_dials.json')
write(OUT / 'completed.json', dict(completed=True, start=START, optimizer_updates=STOP,
      training_ranks=RANKS, segment_ranks=BATCHES[START:STOP], checkpoint=pin(CKPT),
      resumed_from=resume_pin, endpoint=endpoint_pin, dials=dials_pin,
      protocol=pin(protocol_path), trainer=pin(__file__), baseline=pin(SOURCE),
      endpoint_selected=(STOP == 80), native_gradient_certified=False))
print(json.dumps(dict(completed=True, start=START, optimizer_updates=STOP, final_endpoint=STOP==80)), flush=True)
"""
assert 'range(8)' not in s and 'len(RANKS) == len(set(RANKS)) == 32' not in s
assert "p['head'].grad[[0, 15]]" not in s
ast.parse(s)
(D/'train_segment.py').write_text(s)
(D/'native_parameters.py').write_bytes((H/'hill_climb_rrpi_extra32_warm_v1/native_parameters.py').read_bytes())
print(json.dumps(dict(trainer_sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s))))

