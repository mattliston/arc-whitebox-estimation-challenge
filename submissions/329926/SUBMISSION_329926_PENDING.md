# Total-wall repair bank329926 submitted, grade pending

Accepted September5,2026 at14:43:14.020UTC. Phase2 round1429, challenge1174.
This is not a completed official score. Prior329884 remains unusable:
99/100 completed, one TIME_EXHAUSTED on jerry-kerr, aggregate0.01754.

New candidate: candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py
SHA256: 3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757
Archive: fullpair_runtime_rectd4_bulkcopies_v1_whest016.tar.gz
SHA256: 457ebd87de1a131486ebb0fbc0fcad0a0d34e4494948c227b7b03384d326d3cd
All release evidence: verified_local_fullpair_runtime_rectd4_bulkcopies_v1.json.

Repair changes only Winograd rectangle recursion depth5 to4, with smaller
scratch arena. Square schedules, factors, masks, equations and learned
heads remain unchanged. Native35/100/35 head/state gates pass; repeat exact.
Five persistent NumPy-free contexts35/100/500/3016500/35: wall28.685..29.771s
versus36.811..37.744s for329884, mean paired ratio.780177. Residual
.102244..111800s. Sampled combined peak7,455,539,200bytes, below8GiB.
Local macOS/OpenBLAS diagnostics do not guarantee official Linux timings.

Fresh independent4M panel: public8 raw2.45099062e-8, util.2221125,
adjusted5.45083083e-9; virgin6 raw2.00343865e-8, util.2202,
adjusted4.42094383e-9. Combined14 adjusted5.00945069e-9, up7.56% from
4.65737597e-9 because of extra billed compute; raw essentially unchanged.
This is a reliability-bank trade, not progress on the below3e-9 target.

Standard Whest.16 validation, packaging, archive validation all pass;
archive contains only estimator.py and manifest.json. User preauthorized
submission. No forum post, commit, promotion, root-estimator edit, or
credential disclosure. Never rerun the submitted upload script/archive.
Read-only poller: poll_phase2_checkpoint_v4.py. Keep researching raw/util.
