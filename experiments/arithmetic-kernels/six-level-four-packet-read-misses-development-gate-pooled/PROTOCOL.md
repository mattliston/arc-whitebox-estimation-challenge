# Fixed four-packet Read6 experiment

Parent is the fully qualified Weight5 source SHA-256
`a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9`.
Its archive and outstanding approval request remain unchanged.

Implement exactly the layout in `../READ6_PACKET_PREFLIGHT.md`: four packets
2/2/2/1, two Winograd levels below the original D4 leaves,392150528-float
arena,398297088 additional bytes. Original D4/D5 choices remain available;
the shape-only selector adds D6 for m,k,n divisible by64 and uses233 estimated
calls with the existing750000 request penalty. No learned payload, physics,
rank, grouping, selector penalty or amplitude search.

Gate1: actual native unit arithmetic on1024x8192x1024,832x8192x768 and
64x1088x64, plus interleaved repeat. Compare byte-for-byte with the original
fullD6 Winograd product using separately allocated independent scratch frames;
compare to FP64 direct multiplication with relativeRMS<=3e-4. Prove disjoint
live intervals through both child levels and original four backward levels.
Charge cold/warm binds; warm bills must match the exact product formula and
be lower than D5 for both large cases. Record source and resource hashes.

If Gate1 passes, run full native35/100/35 and separate cold100 under the
existing8GB/256MiB-additional-swap guard. Require finite nonnegative outputs,
repeat/warm-cold equality, unchanged first-layer outputs, and measured cold
adjusted ratios against Weight5 <=.99 pooled with both individual ratios<1.
Use the already existing1B development means only after predictions/bills
are sealed. Any gate failure closes this fixed variant; no packet/penalty/
precision search or broader panel follows.

If development passes, real-client35/100/35 must match native outputs/bills,
remain below45s individually, and have pooled wall ratio<=1.10 against Weight5.
Then and only then may the existing14 panel run, with all output/bill records
sealed before its target scoring. Both groups must improve and pooled adjusted
ratio must be<=.99. These are reused panels, not independent validation.

All numerical work is serial, single-threaded and nice10. No process may exceed
the unchanged8GB RSS cap. A resource failure is a measured failure of this fixed
implementation, not permission to raise the cap. Unit reference allocations are
separate per frame so no individual array exceeds the4GB library allocation cap.
No full reference arrays or large matrices are written to disk. No upload is
part of this experiment.
