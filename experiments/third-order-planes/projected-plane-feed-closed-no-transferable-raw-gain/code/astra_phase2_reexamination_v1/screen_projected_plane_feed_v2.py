"""Same fixed physical experiment with selected-support positivity checking."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
runner=HERE/"screen_projected_plane_feed_v1.py"
source=runner.read_text()
source=source.replace("from projected_plane_feed_v1 import predict","from projected_plane_feed_v2 import predict")
source=source.replace('f"projected_plane_feed_r{args.rank}_v1.json"','f"projected_plane_feed_r{args.rank}_v2.json"')
exec(compile(source,str(runner),"exec"),dict(__file__=str(runner),__name__="__main__"))
