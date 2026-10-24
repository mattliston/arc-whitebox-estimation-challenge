# Target-free qualification PASS, September9,2026UTC

Official bank330390 is unchanged at4.53886349950395e-9 adjusted.
This is a harness result, not an accuracy improvement.

Fixed TRAIN35,TRAIN100,TRAIN35 repeat all passed: original d50 all16
served/raw bytes, complete cold bills/operation counts, ordered selectors,
all16 positive prevariances, terminal NumPy replay and exact repeat.
95.8484seconds total; peak sampled combinedRSS6,682,705,920B; zero swap growth.

Qualification result SHA256:
47f417db521120f9462dd069c49720416237a60d956f4a0fd1d2d664f79dd091.
v1 remains failed and unchanged. v2 changes only the offline hidden-buffer
copy plus protocol pins. No native estimator or training objective change.

The fixed TRAIN160 acquisition was then launched under `watch_capture.py`
with this qualification digest. It reuses35/100 and runs158 other fresh
cold children, at most two simultaneously. Outputs are `capture_train_v1`.
No labels may be opened until its manifest and completion seals pass.
The fixed fit and actual native40 objective follow the protocol; no tuning.

Reviewed sources:

- protocol febed4bdccc7c031cb846e6d97f695536ca98d3bef10d07de48f1418a91b6a58
- common52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a
- native744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe
- watch_capture f7354ce9d59945cd02c152716c178656d022d43cec2096b3c4db5530a6b9637a
- fit72b8b688862e93dca28715fab20d63fcab35c1d4a53d1e54bf33db6fcf880007
- watch_fit b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd
