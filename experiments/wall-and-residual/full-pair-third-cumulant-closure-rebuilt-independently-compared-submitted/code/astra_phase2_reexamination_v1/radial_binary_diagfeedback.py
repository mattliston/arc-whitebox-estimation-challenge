"""Two-cube host with mean correction updating covariance diagonals only."""
import inspect
import radial_binary_cubic as binary
source=inspect.getsource(binary.predict)
old='C=C-np.outer(m,dm)-np.outer(dm,m)-np.outer(dm,dm);m=m+dm'
new='C=C-np.diag(2*m*dm+dm*dm);m=m+dm'
assert source.count(old)==1
env=dict(vars(binary));exec(source.replace(old,new),env)
predict=env["predict"]
