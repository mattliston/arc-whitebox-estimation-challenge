# Historical variance supply: compatibility not yet qualified

Phase2,2026-09-09UTC. Read-only source, log, hashes and NPZ scalar/header
metadata audit by root and fresh_accuracy. No numerical moment arrays
decoded, oracle injected, targets scored, labels generated, fit or board
operation. Existing failures and original files remain unchanged.

The often-quoted raw gain is an OLDER rank35 D11/GNN_HV_F1 experiment:
truev_in_loop.sh/.log supplies32M prevariance and scores separate1e9
post means: baseline2.407e-8, half1.826e-8, full3.465e-8.
This is not a d50/current-head result or a panel.
The separate RETRO_SCREEN JSON is an8M noise-corrected frame and must not
be substituted for those endpoints.

Actual force law at fixed mixture s and optional additive noise epsilon:

    v_write = v_truth + s*(v_card-v_truth)
              + epsilon*RMS_layer(v_truth)*xi_i.

The noise is IID by unit, not per-unit relative variance and not halved
when s=.5. The late12..15 old test gives2.040e-8 without noise,
2.572e-8 at epsilon=.001,6.625e-8 at epsilon=.003. This is not a universal
0.1% learned-prediction threshold. Default forcing replaces pre-C diagonal
only, not correlation-preserving covariance congruence.

Original, wider/on-policy, loop-only, coadapted-mean, layer-window and
bilinear variance heads actually failed. Learned a*m.T+m*a.T covariance
also has a completed bad endpoint in chain_acov.log; it was not simply
an interrupted untested fit. Better labels alone are not a new supported
supplier. Independent label noise is not an irreducible prediction floor.

## Artifact identities (G19=public_g19_attribution_v1)

- mps_truth_rank035_P32M.npz and mps_truth_rank0035_P32M.npz are DUPLICATE
  bytes, SHA0876a4a5484259896bdff5d273962dd274c5442e197ccf37bbc274df07c1d193.
- mps_truth_rank0100_P32M.npz SHA9f4576dac4bd435ed0d0016d49d1851065db89084553b56280b894c207d558a2.
- mps_truth_rank0035_indep32M.npz SHAfc0f039cb4f54542c640bb51f72f74a7090cdff2272cd8bcc9552e6fcce807cd.
- mps_truth_rank0100_indep32M.npz SHAabf93fd56d3b405e2d51f253dbaa42e1eedee0801e67754fd27ea0fa5fbfe659.

Scalar-only metadata says P=32,000,000 and expected rank in each. None
embeds RNG seed/XOR, weights SHA, source SHA, command or runtime versions.
Different independent-file member CRCs are not proof of independent draws.

Current generator/STORE mapping resolves to the exact d50/BR weight hashes:
35:14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2;
100:c4923171c8ddef28396aabd70335dad29141361efc13865ccb660a4a57421ee5.
Actual row/weight hashes match STORE. This authenticates current mapping,
not historical execution ownership of every output file.

Current mc_truth_mps.py SHA2a50d16c1ad726ef229d5a2690f6ee597cc78b36ae512875f5949ec3d88b9470;
mc_truth_mps_indep.py SHA5b887ec3ef484f1e715273a375dd4358fa57165f9a955b9061c6c5a228087cb5.
Both use ordinary Gaussian CPU PCG64DXSM input, FP32 MPS products/moment
reductions and FP64 accumulation across batches. The independent script
defaults to the SAME XOR0x47414D454F4E5F31 unless SEED_XOR_OVERRIDE is set.
Located progress/completion logs do not record that environment override.
No contemporaneous pinned acquisition record binding effective seed,
generator, weights and output bytes was found in the bounded audit.

Generation log walls: standard35/100636.5/595.9s, independent35/100
462.1/543.0s. They are historical generation timings, not a promised new
MPS throughput or deployment cost. Logged filenames/counts and mtimes
are consistent with the files; effective independence is not authenticated.

## Decision

No current-d50 variance oracle or head fit admitted. Missing: an authenticated
current-compatible supply/replica uncertainty reference, evidence that its
value survives the current consumer, and an accurate cheap causal supplier.
Old oracle improvement cannot be multiplied by today's cost to advertise
a prospective validated score. New measurements, if later warranted, require
a separate explicit contract; no existing receipt is loosened here.
