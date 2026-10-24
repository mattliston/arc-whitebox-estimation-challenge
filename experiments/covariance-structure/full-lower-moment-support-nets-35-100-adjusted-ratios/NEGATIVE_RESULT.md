# Full lower-moment support: insufficient adjusted improvement

The unchanged candidate improves both development networks but FAILS the
fixed requirement of at least5% adjusted improvement on EACH. It does not
advance to NC14, confirmation, client, package, archive or upload.

| Network | Parent raw MSE | Candidate raw MSE | Candidate cold FLOPs | Raw ratio | Adjusted ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 35 | 1.87060284901e-08 | 1.79839117567e-08 | 410218730507 | 0.961396576844 | 0.990710344893 |
| 100 | 1.84245341753e-08 | 1.72722521502e-08 | 413034826010 | 0.937459367272 | 0.965851998783 |

Pooled adjusted ratio0.9783316543482492 (2.1668% improvement); pooled raw
ratio0.9495187084747343. Complete cold compute rises3.0491% /3.0287%.
Parent raw/adjusted scores reproduce exactly. These are reused development
networks and ordinary1B-reference MSE, not NC14, fresh generalization,
production-client runtime, or an official score.

The original support study failed its first-four-row byte gate. Its source,
protocol, failure, and separate closeout remain unchanged. Source review and
fixed full-width synthetic probes show why equal full support does not imply
exact summation bytes: the compact operator uses score-sorted coordinates,
while the full helpers use natural order. Identity-order helpers match
exactly, permuted helpers do not. This does not attribute every native delta.

This separately pre-target-sealed study used the SAME candidate source
bbae60ddfad2718da44d4bcc6cf50110998675fcc5f7a79e6bc4dd7c5b4b9d12,
SAME saved triple, SAME original5% objective threshold, and one new cold100.
All16 outputs finite/nonnegative FP32; firstTWO parent rows exact; own35
repeat and warm/cold100 exact. All caches retired. Two cold outputs/bills
were sealed before reference means. No coefficient, support, ordering,
threshold, kernel, precision, layer, head or fit changed after the failure.
The older byte gate is not relabeled passed. No objective rescue follows.

Across both studies:7 complete estimator calls, 2781197002080 billed FLOPs;
8 guards (one preserved byte assertion failure), 224.754567s total guard
elapsed, peak RSS6916276224 bytes, zero added swap, all groups retired. Additional
synthetic helper work is separately recorded. No GPU, optimizer updates,
new physical samples or dense weight copies. See resources.json.

Lazy#331599 remains official adjusted3.8164492657795035e-9;4e-9 is banked,
3/2/~1e-9 are unmet and the goal remains active. Qualified Phase2 uploads
remain preauthorized. This is a completed insufficient-improvement result,
not an approval block, a universal impossibility claim, or an unfinished fit.
