# Cost75: shape-only efficiency follow-up to graded Cost150

Official Cost150 #331584 graded4.015535762022727e-9 adjusted,
2.1687335731712666e-8 raw. This is the verified best, but not below4e-9.

Use exactly750,000 estimated FLOPs per remote request in the existing
shape-only kernel selector, instead of1,500,000. This choice is informed by
the already saved shape/cost census, not by new prediction scores. On that
census it saves4.205B arithmetic FLOPs and adds3,747 estimated warm kernel
requests. Physics, coefficients, supported depths, buffers and precision
are unchanged. The archive's zero-penalty extreme adds10,776 requests and
is not selected because platform runtime has caused actual past failures.

First run cold35 and100 plus repeat35. Require finite nonnegative16x1024
outputs, exact repeat, lower measured complete bill, and improved adjusted
error on BOTH fixed development networks using their existing1e9 truths.
Cold100 must be measured separately; warm bills cannot stand in for cold.
If this fails, preserve the negative without tuning penalties on those
outputs. If it passes, run the unchanged public8+fresh6 panel, seal all
outputs and bills before scoring, then measure the real NumPy-free client
boundary with35,100,35. Runtime increase must be explicitly reported.

Use the same qualification standard as Cost150: lower aggregate adjusted
noise-corrected error on each panel, no large isolated regression, and
production-boundary exact parity/repeat. Package and submit if qualified
and the local ratio to the graded parent projects below4e-9. Projection
is not an official result. Preserve the graded parent and all earlier work.
No parallel numerical workers; existing8GB/swap/time guards apply.
