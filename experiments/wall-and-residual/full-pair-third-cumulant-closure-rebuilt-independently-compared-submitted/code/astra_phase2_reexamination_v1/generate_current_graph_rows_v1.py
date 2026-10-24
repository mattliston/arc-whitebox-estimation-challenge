"""Fresh graph rows with byte-exact host and old-feature parity assertions."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE / "generate_radial_rows.py"
source = path.read_text()
edits = {
    "from k3_radial_numpy import predict":
        "from radial_compact_feed_fullpair_v1 import predict as base_predict\n"
        "def predict(W, **kwargs):\n    return base_predict(W, gate_floor=.01, **kwargs)",
    "from radial_features import features,FEATURE_NAMES":
        "from radial_graph_features_v1 import features,FEATURE_NAMES",
    "np.save(out/\"raw20.npy\",x[:,:,:20]); np.save(out/\"mech.npy\",x[:,:,20:])":
        "parent=HERE/'rows_compact_feed_fullpair_v1'/f'row_{args.rank:04d}'\n"
        "assert np.array_equal(host,np.load(parent/'host.npy')),('host drift',args.rank)\n"
        "assert np.array_equal(sigma,np.load(parent/'sigma.npy')),('sigma drift',args.rank)\n"
        "assert np.array_equal(x[:,:,:20],np.load(parent/'raw20.npy')),('raw20 drift',args.rank)\n"
        "assert np.array_equal(x[:,:,20:30],np.load(parent/'mech.npy')),('old mech drift',args.rank)\n"
        "assert x.shape==(16,1024,50)\n"
        "np.save(out/\"raw20.npy\",x[:,:,:20]); np.save(out/\"mech.npy\",x[:,:,20:])",
    "result = dict(rank=args.rank,args=vars(args),":
        "result = dict(parent_host_sigma_features_byte_exact=True,rank=args.rank,args=vars(args),",
}
for old, new in edits.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(compile(source, str(path), "exec"), dict(__file__=str(path), __name__="__main__"))
