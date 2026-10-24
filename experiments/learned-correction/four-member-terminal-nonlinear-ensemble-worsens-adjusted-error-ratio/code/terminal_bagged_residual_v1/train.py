"""One fixed five-fold, four-member terminal residual fit; no checkpoint choice."""
import hashlib
import json
import math
from pathlib import Path
import resource
import time

from model import ResidualMLP, control_fit, numpy_forward, project_torch, torch, np

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
CAPTURE = D / 'terminal_representation_v1'
READ5_SHA = 'd7723068ae269155e0369e4ba322b447222bbb31295d89394c069695ad0e824b'
MANIFEST_SHA = '5c49e3f22217c597b0702497da93e5d7d034d115de762a00c9923ee2fb94e339'
TARGET_MANIFEST_SHA = 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
TRAIN = [35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58]
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def finite_net(net, optimizer):
    for parameter in net.parameters():
        assert parameter.device.type == 'cpu' and parameter.dtype == torch.float32
        assert torch.isfinite(parameter).all().item()
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all().item()
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all().item()


def main():
    started = time.monotonic()
    certificate = read(HERE / 'algebra_certificate.json')
    assert certificate['passed'] and certificate['model_sha256'] == sha(HERE / 'model.py')
    watcher = read(HERE / 'certificate_watch.json')
    assert watcher['completed'] and watcher['whole_group_retired']
    assert sha(CAPTURE / 'train_capture/manifest.json') == MANIFEST_SHA
    manifest = read(CAPTURE / 'train_capture/manifest.json')
    assert manifest['completed'] and not manifest['truth_accessed'] and manifest['train'] == TRAIN
    assert manifest['capture_source_sha256'] == sha(CAPTURE / 'capture.py')
    assert manifest['protocol_sha256'] == sha(CAPTURE / 'PROTOCOL.md')
    out = HERE / 'training_v1'
    out.mkdir(exist_ok=False)
    data = []
    for ordinal, row in enumerate(manifest['rows']):
        record = row['record']
        watch = read(row['watch'])
        assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes'] == 0
        assert row['rank'] == TRAIN[ordinal] and row['ordinal'] == ordinal
        assert record['candidate_sha256'] == READ5_SHA and record['includes_initialization']
        assert sha(row['completion']) == row['completion_sha256']
        assert sha(record['capture']) == record['capture_sha256']
        with np.load(record['capture']) as file:
            arrays = {key: file[key].astype(np.float64) for key in ('prediction', 'sigma', 'hidden', 'input')}
        assert arrays['input'].shape == (1024, 126) and arrays['hidden'].shape == (1024, 64)
        assert all(np.isfinite(a).all() for a in arrays.values())
        assert np.all(arrays['sigma'] > 0)
        x = arrays['sigma'][:, None] * np.concatenate([np.ones((1024, 1)), arrays['hidden']], axis=1)
        data.append(dict(row=row, prediction=arrays['prediction'][-1], sigma=arrays['sigma'],
                         control=x, inputs=arrays['input']))

    # All outer folds, bootstrap network draws and initial arrays precede target reads.
    seeds = []
    for fold in range(5):
        train = [i for i in range(20) if i % 5 != fold]
        held = [i for i in range(20) if i % 5 == fold]
        for member in range(4):
            init_seed = 190919700 + 16 * fold + member
            bootstrap_seed = 190919900 + 16 * fold + member
            rng = np.random.Generator(np.random.PCG64DXSM(init_seed))
            payload = dict(w=rng.uniform(-1 / np.sqrt(126), 1 / np.sqrt(126), size=(32, 126)).astype(np.float32),
                           b=np.zeros(32, np.float32), v=np.zeros(32, np.float32), c=np.array(0, np.float32))
            init_path = out / f'fold{fold}_member{member}_initial.npz'
            np.savez(init_path, **payload)
            drawn = np.random.Generator(np.random.PCG64DXSM(bootstrap_seed)).integers(0, 16, size=16)
            counts = np.bincount(drawn, minlength=16)
            seeds.append(dict(fold=fold, member=member, train_ordinals=train, held_ordinals=held,
                              initialization_seed=init_seed, bootstrap_seed=bootstrap_seed,
                              bootstrap_draws=drawn.tolist(), bootstrap_counts=counts.tolist(),
                              initialization=str(init_path), initialization_sha256=sha(init_path)))
    target_manifest = P2 / 'astra_phase2_reexamination_v1/joint_prefix_final_fit_v1/manifest.json'
    assert sha(target_manifest) == TARGET_MANIFEST_SHA
    target_sources = {r['rank']: r for r in read(target_manifest)['target_sources']}
    with (out / 'inputs_sealed.json').open('x') as stream:
        json.dump(dict(source_sha256=sha(__file__), model_sha256=sha(HERE / 'model.py'),
                       protocol_sha256=sha(HERE / 'PROTOCOL.md'), capture_manifest_sha256=MANIFEST_SHA,
                       target_manifest_sha256=TARGET_MANIFEST_SHA, target_files_opened=False,
                       ranks=TRAIN, fixed_fits=seeds), stream, indent=2)
    target_hashes = {}
    for datum in data:
        rank = datum['row']['rank']
        path = P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz'
        binding = target_sources[rank]
        assert str(path) == binding['target_file'] and sha(path) == binding['target_sha256']
        assert sha(binding['weight_file']) == datum['row']['record']['weights_sha256']
        with np.load(path) as file:
            assert file['mc_post'].dtype == np.float32
            truth = file['mc_post'][-1].astype(np.float64)
        datum.update(truth=truth, residual=truth - datum['prediction'])
        target_hashes[str(rank)] = sha(path)

    oof_control = np.empty((20, 1024), np.float32)
    oof_ensemble = np.empty_like(oof_control)
    oof_members = np.empty((4, 20, 1024), np.float32)
    fold_records = []
    total_updates = 0
    for fold in range(5):
        fold_started = time.monotonic()
        train = [i for i in range(20) if i % 5 != fold]
        held = [i for i in range(20) if i % 5 == fold]
        cat = lambda key, ids: np.concatenate([data[i][key] for i in ids])
        weight = np.concatenate([np.full(1024, data[i]['row']['record']['parent_multiplier']) for i in train])
        x, inputs = cat('control', train), cat('inputs', train)
        sigma, prediction, truth = cat('sigma', train), cat('prediction', train), cat('truth', train)
        control = control_fit(x, truth - prediction, weight)
        assert control['rank'] == 65
        with np.load(CAPTURE / 'information' / f'fold{fold}_fit.npz') as old:
            old_prediction = (x / old['sx']) @ old['beta']
        control_error = float(np.max(np.abs(control['xn'] @ control['beta'] - old_prediction)))
        assert control_error < 1e-10, control_error
        center = np.sum(inputs * weight[:, None], axis=0) / weight.sum()
        scale = np.maximum(np.sqrt(np.sum((inputs - center)**2 * weight[:, None], axis=0) / weight.sum()), 1e-6)
        normalized = np.clip((inputs - center) / scale, -8, 8).astype(np.float32)
        base = prediction + control['xn'] @ control['beta']
        r = np.float32(max(2**-20, np.sqrt(np.sum(weight * ((truth - base) / np.maximum(sigma, 1e-12))**2) / weight.sum())))
        xn_held = cat('control', held) / control['scale']
        base_held = cat('prediction', held) + xn_held @ control['beta']
        input_held = np.clip((cat('inputs', held) - center) / scale, -8, 8).astype(np.float32)
        sigma_held = cat('sigma', held)
        xt = torch.tensor(normalized, dtype=torch.float32)
        ht = torch.tensor(input_held, dtype=torch.float32)
        xnt = torch.tensor(control['xn'], dtype=torch.float64)
        pinvt = torch.tensor(control['pinv'], dtype=torch.float64)
        baset = torch.tensor(base, dtype=torch.float64)
        targett = torch.tensor(truth, dtype=torch.float64)
        rawscale = torch.tensor((sigma * r).astype(np.float32))
        heldscale = torch.tensor((sigma_held * r).astype(np.float32))
        held_corrections = []
        member_records = []
        for member in range(4):
            member_started = time.monotonic()
            fixed = seeds[4 * fold + member]
            assert sha(fixed['initialization']) == fixed['initialization_sha256']
            with np.load(fixed['initialization']) as file:
                payload = {key: file[key] for key in file.files}
            net = ResidualMLP(payload)
            assert sum(v.numel() for v in net.parameters()) == 4097
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8,
                                           weight_decay=2e-4, foreach=False)
            boot = weight * np.repeat(np.array(fixed['bootstrap_counts']), 1024)
            wt = torch.tensor(boot / boot.sum(), dtype=torch.float64)
            initial_served = torch.clamp(baset, min=0).to(torch.float32).to(torch.float64)
            denominator = torch.sum(wt * (initial_served - targett)**2).detach()
            assert float(denominator) > 0
            trace = []
            for step in range(200):
                lr = 1e-4 + .5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * step / 199))
                optimizer.param_groups[0]['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                q = rawscale * net(xt)
                assert torch.isfinite(q).all().item()
                delta, coefficient = project_torch(q, xnt, pinvt)
                served = torch.clamp(baset + delta, min=0).to(torch.float32).to(torch.float64)
                loss = torch.sum(wt * (served - targett)**2) / denominator
                assert torch.isfinite(loss).item()
                loss.backward()
                finite_net(net, optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                finite_net(net, optimizer)
                total_updates += 1
                if step % 20 == 0 or step == 199:
                    observation = dict(update=step + 1, normalized_bootstrap_loss=float(loss.detach()),
                                       lr=lr, preclip_grad_norm=float(grad_norm))
                    trace.append(observation)
                    print(json.dumps(dict(fold=fold, member=member, **observation,
                                          elapsed_s=time.monotonic() - started)), flush=True)
            assert all(int(state['step'].item()) == 200 for state in optimizer.state.values())
            payload = net.export()
            payload_path = out / f'fold{fold}_member{member}_terminal.npz'
            np.savez(payload_path, **payload)
            del optimizer, net, q, delta, coefficient, served, loss
            with np.load(payload_path) as file:
                reloaded = {key: file[key] for key in file.files}
            assert all(np.array_equal(payload[key], reloaded[key]) for key in payload)
            net = ResidualMLP(reloaded)
            for parameter in net.parameters():
                parameter.requires_grad_(False)
            with torch.no_grad():
                q_train = rawscale * net(xt)
                delta_train, coeff = project_torch(q_train, xnt, pinvt)
                q_held = heldscale * net(ht)
                train_served = torch.clamp(baset + delta_train, min=0).to(torch.float32).to(torch.float64)
                train_raw = float(torch.sum(torch.tensor(weight / weight.sum()) * (train_served - targett)**2))
            qh = q_held.numpy().astype(np.float64)
            coefficient = coeff.numpy().copy()
            delta_held = qh - xn_held @ coefficient
            assert np.isfinite(delta_held).all()
            held_corrections.append(delta_held)
            orthogonality = float(np.max(np.abs(control['xn'].T @ (weight * delta_train.numpy()))) / weight.sum())
            assert orthogonality < 1e-10
            numpy_q = (sigma_held * r).astype(np.float32) * numpy_forward(reloaded, input_held)
            forward_gap = float(np.max(np.abs(numpy_q.astype(np.float64) - qh)))
            assert forward_gap <= max(1e-10, float(np.max(np.abs(qh))) * 1e-5)
            coeff_path = out / f'fold{fold}_member{member}_projection.npy'
            np.save(coeff_path, coefficient)
            member_record = dict(member=member, terminal_payload_sha256=sha(payload_path),
                                 projection_sha256=sha(coeff_path), updates=200,
                                 training_raw=train_raw, weighted_orthogonality=orthogonality,
                                 torch_numpy_raw_correction_max_abs=forward_gap,
                                 elapsed_s=time.monotonic() - member_started, trace=trace)
            member_records.append(member_record)
            with (out / f'fold{fold}_member{member}_trace.json').open('x') as stream:
                json.dump(member_record, stream, indent=2)
            del net, q_train, delta_train, coeff, q_held, train_served, wt
        corrections = np.stack(held_corrections)
        ensemble_prefloor = base_held + corrections.mean(axis=0)
        members_prefloor = base_held[None, :] + corrections
        for slot, ordinal in enumerate(held):
            sl = slice(slot * 1024, (slot + 1) * 1024)
            oof_control[ordinal] = np.maximum(base_held[sl], 0).astype(np.float32)
            oof_ensemble[ordinal] = np.maximum(ensemble_prefloor[sl], 0).astype(np.float32)
            oof_members[:, ordinal] = np.maximum(members_prefloor[:, sl], 0).astype(np.float32)
        fit_path = out / f'fold{fold}_shared.npz'
        np.savez(fit_path, center=center, scale=scale, residual_scale=r,
                 control_scale=control['scale'], control_beta=control['beta'],
                 singular=control['singular'])
        # Store raw OOF predictions for the later diagnostic; no held-out scoring yet.
        fold_path = out / f'fold{fold}_held_predictions.npz'
        np.savez(fold_path, control=oof_control[held], ensemble=oof_ensemble[held],
                 members=oof_members[:, held], members_prefloor=members_prefloor,
                 ensemble_prefloor=ensemble_prefloor)
        fold_record = dict(fold=fold, train_ranks=[TRAIN[i] for i in train],
                           held_ranks=[TRAIN[i] for i in held], control_rank=control['rank'],
                           control_recompute_max_abs=control_error, shared_sha256=sha(fit_path),
                           held_predictions_sha256=sha(fold_path), members=member_records,
                           elapsed_s=time.monotonic() - fold_started)
        fold_records.append(fold_record)
        with (out / f'fold{fold}_sealed.json').open('x') as stream:
            json.dump(fold_record, stream, indent=2)
        print(json.dumps(dict(fold=fold, held_predictions_sealed=True,
                              elapsed_s=time.monotonic() - started)), flush=True)
        del xt, ht, xnt, pinvt, baset, targett, rawscale, heldscale

    assert total_updates == 4000
    assert all(np.isfinite(a).all() and np.all(a >= 0) for a in (oof_control, oof_ensemble, oof_members))
    all_path = out / 'oof_predictions.npz'
    np.savez(all_path, control=oof_control, ensemble=oof_ensemble, members=oof_members)
    with (out / 'oof_sealed.json').open('x') as stream:
        json.dump(dict(predictions_sha256=sha(all_path), folds=fold_records,
                       target_sha256=target_hashes, total_updates=4000, fits=20,
                       selection_used=False, all_predictions_precede_aggregate_scoring=True), stream, indent=2)
    rows = []
    for ordinal, datum in enumerate(data):
        row = dict(rank=TRAIN[ordinal], fold=ordinal % 5)
        parent_flops = datum['row']['record']['parent_flops']
        for name, prediction, reserve in [
            ('unchanged', datum['prediction'], 0), ('control', oof_control[ordinal], 1_000_000),
            ('ensemble', oof_ensemble[ordinal], 80_000_000)]:
            raw = float(np.mean((prediction.astype(np.float64) - datum['truth'])**2))
            bill = parent_flops + reserve
            row[name] = dict(raw=raw, flops=bill, adjusted=raw * max(.1, bill / 2**41))
        row['members'] = []
        for member in range(4):
            raw = float(np.mean((oof_members[member, ordinal].astype(np.float64) - datum['truth'])**2))
            bill = parent_flops + 20_000_000
            row['members'].append(dict(raw=raw, flops=bill, adjusted=raw * max(.1, bill / 2**41)))
        rows.append(row)
    arms = ('unchanged', 'control', 'ensemble')
    pooled = {arm: float(np.mean([row[arm]['adjusted'] for row in rows])) for arm in arms}
    folds = [{arm: float(np.mean([row[arm]['adjusted'] for row in rows if row['fold'] == fold]))
              for arm in arms} for fold in range(5)]
    wins = sum(row['ensemble'] < min(row['unchanged'], row['control']) for row in folds)
    identity = []
    for fold in range(5):
        held = [i for i in range(20) if i % 5 == fold]
        truth = np.concatenate([data[i]['truth'] for i in held])
        with np.load(out / f'fold{fold}_held_predictions.npz') as file:
            members = file['members_prefloor']; ensemble = file['ensemble_prefloor']
        mean_individual = float(np.mean((members - truth)**2))
        spread = float(np.mean(np.var(members, axis=0)))
        ensemble_raw = float(np.mean((ensemble - truth)**2))
        gap = abs(ensemble_raw - (mean_individual - spread))
        assert gap < 1e-18
        identity.append(dict(fold=fold, mean_individual_unclipped_raw=mean_individual,
                              member_spread=spread, ensemble_unclipped_raw=ensemble_raw, identity_abs_error=gap))
    passed = pooled['ensemble'] <= .95 * pooled['unchanged'] and pooled['ensemble'] <= .97 * pooled['control'] and wins >= 4
    result = dict(schema='terminal_bagged_residual_information_v1', exploratory_network_cv=True,
                  new_backbone_calls=0, training_updates=4000, models=20, parameters_per_model=4097,
                  rows=rows, pooled_adjusted=pooled, fold_adjusted=folds, fold_wins=wins,
                  ensemble_to_unchanged=pooled['ensemble'] / pooled['unchanged'],
                  ensemble_to_control=pooled['ensemble'] / pooled['control'],
                  mean_single_member_adjusted=float(np.mean([m['adjusted'] for row in rows for m in row['members']])),
                  ensemble_variance_diagnostic=identity, gate_passed=bool(passed),
                  source_sha256=sha(__file__), model_sha256=sha(HERE / 'model.py'),
                  protocol_sha256=sha(HERE / 'PROTOCOL.md'), capture_manifest_sha256=MANIFEST_SHA,
                  numpy_version=np.__version__, torch_version=torch.__version__, device='cpu',
                  threads=torch.get_num_threads(), deployment_bill_qualified=False,
                  elapsed_s=time.monotonic() - started,
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (out / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('rows', 'fold_adjusted', 'ensemble_variance_diagnostic')}), flush=True)


if __name__ == '__main__':
    main()
