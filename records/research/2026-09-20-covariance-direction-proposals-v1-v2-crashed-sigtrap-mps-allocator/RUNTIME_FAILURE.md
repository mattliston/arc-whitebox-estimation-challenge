# Version1 runtime failure, no scientific endpoint

The proposal process29086 terminated SIGTRAP after10.0917s, before any graph
receipt or direction was saved. Its proposal directory existed, so the
first FIT35 target may already have been read; it is false to call this a
pre-import failure. No complete backward, parameter update, native model
call or score result was produced. Unit checks had passed exactly.

The matching macOS crash report identifies a libdispatch same-queue sync
trap while the MPS allocator released cached buffers inside an indexing
kernel. This identifies the crashing call chain, not a proven memory demand
or general allocator cause. RSS peak1.028GB, zero added swap; group retired.

Version2 is a separately sealed runtime-only recovery: keep4.5GB MPS and
8GB RSS caps, but synchronize and empty the device cache after each complete
step outside a kernel; record per-step allocation and enable faulthandler.
All parameters, action, four FIT networks, radius, loss and native scientific
gates remain fixed. The discarded first partial forward supplies no proposal
or tuning feedback. No package/library/installed-source modifications.

PyTorch2.7 documents the hard allocation limit and the low watermark's
cache-collection/adaptive-commit role. This does not prove the crash cause:
https://docs.pytorch.org/docs/2.7/mps_environment_variables.html
https://docs.pytorch.org/docs/2.7/generated/torch.mps.set_per_process_memory_fraction.html
