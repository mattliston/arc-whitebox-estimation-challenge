# BEGIN FABLE-ENGINEER TREE-CACHE CAP RAISE 480 (E-75a, 2026-09-25; bit-identical): the alt-basis tree caches' LRU caps go from 160 to 480
# per side. Each net binds 12-80 net-specific tree keys; at 160 per side the caches saturate by about the 4th net of a worker, and
# from then on keys that earlier nets used are evicted and bound again inside predict, billed at numel (e75_lazybind_census:
# 15-34 rebinds per net on nets 5-8, 0.14-0.44% of the bill, growing). Cached entries are views of the arena, so predictions are
# unchanged; the cost is live server handles (views), gated by the handle census.
_CAP_BY_ID[id(_IPT_A)] = 480; _CAP_BY_ID[id(_IPT_B)] = 480; _CAP_BY_ID[id(_IPT_C)] = 480
# END FABLE-ENGINEER TREE-CACHE CAP RAISE 480
