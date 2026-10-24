# Q24 CP3 chronological age-retirement discriminator

Status: **FAIL — blind chronological age retirement is terminal.**

Frozen scope: observer-fit rank 0 only; no fitting, observer selection, network
access, or submission.  Every arm preserves the frozen native-J logical
full-birth rank feature.  At each retirement boundary the full carried CP3
state is used for the current K21 reset; only the post-reset future carrier is
shortened, and the dense cubic star absorbs the retired K21 discrepancy.

| arm | physical CP3 rank | host final MSE | native-J final MSE | mean q10:15 gap MSE | exact-q10:15 suffix oracle | literal prototype FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| full | growing to 15360 | 8.212858909e-8 | 2.856783180e-8 | 5.384855913e-9 | 5.362486676e-9 | 1,574,161,982,840 |
| w1 | 1024 | 7.132925829e-7 | 6.351812535e-7 | 1.113382520e-7 | 2.011786289e-7 | 838,922,001,812 |
| w2 | 2048 | 4.964476460e-7 | 4.166148553e-7 | 7.640948263e-8 | 8.615974104e-8 | 944,602,732,920 |
| w3 | 3072 | 3.768622419e-7 | 2.731971975e-7 | 5.290833710e-8 | 4.125684750e-8 | 1,042,663,998,840 |
| w4 | 4096 | 2.560825296e-7 | 1.647952764e-7 | 3.569327907e-8 | 1.615336780e-8 | 1,133,179,185,528 |
| w6 | 6144 | 1.480227802e-7 | 8.140125995e-8 | 1.681617980e-8 | 6.694549996e-9 | 1,291,571,321,208 |

The native-J loss relative to full is 22.23x, 14.58x, 9.56x, 5.77x, and
2.85x for w1, w2, w3, w4, and w6 respectively.  The first material divergence
always appears one affine layer after the first retirement: zero-based layers
2, 3, 4, 5, and 7.  This localizes the failure to future transport of retired
triple-distinct cubic content.  It is not deletion at the current reset: the
split retire+keep read agrees with the unsplit current-reset K21 to roughly
6.0--6.9e-8 maximum absolute / 6.4--7.0e-9 RMS (and exactly for w1).

The w6 exact-q10:15 suffix oracle remains 6.695e-9, close to the full-host
5.362e-9 oracle, while its autonomous final error is 8.140e-8.  Thus old
columns cannot be selected by age alone; any smaller carrier must retain the
old subspace that acts on later layers.  The only justified successor is an
equal-rank selector (fixed earliest prefix, tensor energy, or a deployable
target-free downstream-action score), with the same exact current-K21 star
residual and logical native-J rank features.

Runner SHA-256:
`a00030b8f5f5b0cada668eedd6c30971426aa50692276c4bf911387d4fed9592`.

Per-arm JSON SHA-256 values:

- full: `05201cae91be1b368f5dfe005fb68378d8cc2728264f5e684a04d7a0e6e64e2f`
- w1: `12f53299cc824dbdb3e1df2c7b1c50d664506e4d23069914ab2515e87dc95a1e`
- w2: `aa78d707e194087df0c955e88d800f019f6431577b66d00a58081bb2d74eaa24`
- w3: `fc8d13897338a5430dbd6f3e889afcaaf2faa7e450181296b3b9a5ca9aa86049`
- w4: `1708b1ec126305119ce7927c3c55efc4cf418edc430eaa346d29682b2249ee29`
- w6: `3b66a53d7e0734a9a005de6ca0670669ddeed2a755bfe02b955353cb319d66f6`
