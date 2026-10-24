# Trainer handoff to Opus 5.5

Written 2026-10-11T03:30:06Z from `date -u`. This lane has reached the user-requested pause boundary. No trainer process remains live according to its completed guard; root will independently check PIDs. The background A30 producer is deliberately left to root.

## Final Nc decision: NOT ACCEPTED

The exact96-gradient, one-point cut-simulated refit ended **2026-10-11T03:27:57Z**. It used all12 predeclared live arms,192 sizing arm evaluations, and exactly one fixed point. There were352 unique completed requests:96 public gradients and256 evaluations. No worker error, recycle, or retry occurred.

- Sizing136–151: **−0.9117023509953137%**, SE0.3455137577220325%,10/16 better; all-layer mean−0.4139880502656679%.
- Independent check152–167: **−0.2354527961318409%**, SE0.3403561212510006%,10/16 better; all-layer mean−0.31765381740289644%.
- Fixed acceptance rule: pooled change_pct < −SE of per-network percentage changes. The check improvement was smaller than its SE, so the point was rejected. These sets are now selection data, not untouched promotion evidence.
- Solver forecast was−0.9880074545862261%, LOO−0.3959370164337228%; the actual finite check above controls the decision.
- No export, final judge, native build, release, submission, next point, or EDBAND experiment was launched. **A27 final168–199 remains untouched/sealed**; all reference files being ready does not authorize reading them for this rejected point.

**Standing best remains S46:** `cert/refit/jmI_p2_noAS.pt` SHA `f36e034c6196f7693b4e01c453f7e2044aca9b69010bd2f4ab5c2340022411e3`, head `cert/refit/jmI_p2_hu.json` SHA `d5b16582cfec6f84ae3f50d75e3d854c91585da5284a83f3742a45af7e40fa7c`. The rejected trial is preserved at `cert/refit/jmNc_p1.pt` SHA `42510c2780e507a7e353d9b121008508314a0dc13f6eead691286c10aaed215b`, head `cert/refit/jmNc_p1_hu.json` SHA `12b046f948473bb8ca0e15604d06475d82803591f7ac3b8895fac3f693ca48d8`. Do not promote or export the rejected trial, weaken the rule, resize it from this check, or run another Nc point using these sets as fresh evidence.

Authoritative decision: [jmNc_first_point.json](cert/refit/jmNc_first_point.json), SHA `9460962f9a797aa5a3c6dadc0301c1447bf16106520757db6c009b12d707e549`. Its full request, all source/checkpoint/rank pins, solve and all selection statistics are retained. [FINAL_RECEIPT_v1.json](cert/cut_refit_Nc_launch_v1/FINAL_RECEIPT_v1.json), SHA `9614db536f412832c15be81aa30256b1e1f53756584d5fcde3c2a7dcb84251b6`, binds the final decision, complete process receipts, immutable logs and current source-ready artifacts.

## Process and resource state

Driver26556 exited0 at03:27:57Z. Guard26404 exited0 at03:27:58Z; its worker26407 record says completed=true, exit_code0, error=null, whole_group_retired=true, remaining_rows=[]. Tool sessions19567(driver) and72882(worker/guard) both returned exit0 and are closed.

Guard peakRSS **8,838,987,776 bytes**, peak footprint1,864,440,504 bytes, swap growth0. Elapsed10405.706s (~2h53m26s). It used one8-thread worker; actual hard RSS backstop was10,000,000,000 bytes, footprint trigger7,500,000,000 bytes. Do not confuse the planned9GB reservation with that actual10GB backstop.

`run_M/STOP` was created03:27:57Z and must remain. For any later authorized resumption, version/archive it, never delete or overwrite. The older STOP was already archived at `run_M/STOP.archived_20261011T003427Z_before_Nc`. Do not restart the old supervisor91436/workers91458/91554 or duplicate the finished Nc driver. No auto-restart or second worker is configured or authorized.

Receipts/logs: `cert/cut_refit_Nc_launch_v1/{driver_terminal.json,worker_terminal.json,worker_guard.json,driver_handle.json,worker_handle.json,prelaunch.json,driver.log,worker.log}`. Exact launch was `/Users/matt/anaconda3/bin/python -B cert/launch_cut_firstpoint_v2.py` with its recorded role/threads8/directory arguments. The driver argv was `bash cert/refit/jmix24.sh Nc cert/refit/jmI_p2_noAS.pt cert/refit/jmI_p2_hu.json 1 1 0 '[{"module":"pbase","bsk":0.3402257055212658}]' cert/refit/gen_size16_v4.txt`; use recorded handle/guard environment as authority, not an improvised relaunch.

## What Opus should do next

1. Read root's final `ASTRA_TO_OPUS_HANDOFF_20261011...` master and reconcile the live A30 producer/status before any compute. Root launched A30v2 PID17133 at02:49:01Z; trainer did not touch it. Its plan is `analyst/a30_edband_selection_and_final_20261011T0057Z/plan_v2.json`, SHA `b35e8a1c7c83d4018ea44b8b01464f96069cab02f1583f2e0bfbfbc9ee758f4a`. New sizing232–247/check248–263, then final200–231; all15,990,784 samples. Root controls candidate assignment, safe resource gates, and clean network-boundary STOP for quiet release timing.
2. Start the next scientific direction from **standing S46**, not rejectedNc. The prepared next direction is EDGE shared/band bias/output, with exactly24 public gradients (first24 of the prescribed Nc96 order) and at most6 identifiable directions. Use genuinely fresh A30 selection sets. No extra work is authorized while paused; obtain the resumed lead's runtime/protocol/resource decision.
3. Before that24-gradient experiment, complete genuine full-network zero-offset identity against standing S46, fixed nonzero EDBAND routing/native parity, and normal production export/build evidence. These are real unfinished runtime gates. Source review or the current v11 S46 point parity does not satisfy them.
4. The optional historical used651 pair was **PARKED, never run** because no natural22GiB concurrent-memory window appeared. Prepared `cert/edband_used651_pair_v1.py` and parking receipt remain. With Nc retired, reassess resources rather than blindly reuse the old concurrent gate. A historical zero-only pair can provide a real structural export result, but cannot by itself qualify all latest-best/nonzero routing gates.
5. If the future EDBAND point is accepted, use the appropriate envelope/export/judge path and newly assigned untouched final block. Do not silently drop its sidecar into the plain Nc exporter. Native build and qualification must retain full16 DLY and exact HU/source identities, safe+twin, no underpriced operations. Only fire/fire_once.py submits; never read/copy key.txt.

## Source-ready EDBAND: final versions only

- `edband_six_driver_v6.py` SHA `8e3dc70a9ff66de70f59a3687fbd2da65b908ea20bf077f67d3967357c93caf9`.
- `edband_qualification_v6.py` SHA `91b06e8cdc7701bfb0b587eb305f60fd16c2370b74144e30d2ae73e262829ba0`.
- `worker_edband_v2.py` SHA `f1dae5b23057a08871e9691fbb42e0cbe7b0565e1d970c6eea273ed66618366f` and `export_edband_s46_v2.py` SHA `6cb291bc3e690178386e68f9b3cd4fa22937e1b77a226fd965e8be0337ed8978`.
- Full inventory/config guide: [source freeze](cert/edband_driver_source_freeze_v1/RECEIPT.json) and [README](cert/edband_driver_source_freeze_v1/README.txt). Independent final review is `physicist/T90_EDBAND_DRIVER_SCORER_REVIEW_v1.json` SHA `ef8368705a35f70d498f05606b5d9bfcabcf0f296eff3b5df599b125a240c076`.
- Driver queues exact24 eligible public gradients, <=6 arms, fresh16 sizing, one fixed point, independent16 check under the same pooled-change/one-SE convention, then STOP. Public gradients can start before truth completion; generated requests must await immutable completion receipts. Final32 remains sealed. It never self-launches a worker or final judge and has no automatic resume/next point.
- Qualification requires retrievable actual plain/enabled requests/results and exact all16 double-valued prediction/MSE identity; full checkpoint/base/payload/request/runtime binding; native source/export/build/guard/NPY/report evidence at651+378 with recomputed finalRMS<=7e−6; and a genuinely fixed-nonzero probe. Boolean stubs do not pass. Earlier v1–v5 drafts are preserved but not current.
- Normal exporter requires genuine result metadata. **Do not fabricate a neural result or create a fixture-only schema/adapter**; root explicitly rejected that workaround. Raw native parent must equal the actual base export, not an arbitrary transformed SHA.
- Engineer combined raw-parent builder is `engineer/a56_edge_band/edband_s46_adapter_v3.py`; its real combined text build is still pending a genuine COMPLETE bundle. Native evidence contract is `engineer/a56_edge_band/NATIVE_PARITY_CONTRACT_v1.json` SHA `e56ecad2820ced16a2db419ff175fd81270130ba251a8a0995d0ab08931debbe`. Tiny stdlib/source fixtures passed, not a native full-card run.

## Certified foundation and limits

The production simulator remains the exact frozen S46 v11 profile `engineer/a22_cutsim/S46_v11_PINNED_PROFILE.json` SHA `06fe3a1b601a7893f61725a8ddee17a590039ab21aaaf7477385b466890a2f7b`, with enabled, law_response_fit, law_beta and law_restore_right all true and all dependency hashes pinned. [Scoped certificate](cert/cutsim_s46_20261010T232518Z/CERTIFICATE_V11_SCOPED_v1.json) records651/378 fidelity, exact gradient/eval predictions, ordered replay, finite parameter gradients, and bounded local cut VJP checks. Whole-card EDGE AD/FD still has documented inherited frozen-carrier/head surrogate discrepancies; do not claim exact full-network derivatives. The8-thread probe was separately checked and accepted for tiny rounding/native residual.

Keep current nonzero HU rows6/17/20/21/28/29 only. Dormant feature columns and future repair HU31/32 are not covered by this certificate. Head128→256 and auxloss are closed; repair-head/EDBAND source scaffolding is not production qualification. Old standalone cutsimv2, original stale trainer drivers, and endpoint-dropping DLY converterv1 must not replace these pins. Use engineer `a52_pld/dly_from_trainer_v2.py` for full0–15 absolute sidecars when a real accepted point exists.

All earlier experiments, certificate variants and source preparations remain archived; this note intentionally focuses on the final Nc outcome and next trainer branch. The root handoff indexes broader results. No descendants were spawned or restarted for this handback.
