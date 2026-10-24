"""Own the isolated OpenBLAS diagnostic server and sample combined RSS."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time

HERE=Path(__file__).resolve().parent
GATE=HERE.parent/"data_gate"
ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--tag",required=True)
args=ap.parse_args()
out=HERE/f"{args.tag}_resources.json"
log=HERE/f"{args.tag}_server.log"
assert not out.exists() and not log.exists() and not (HERE/f"{args.tag}.json").exists()
scratch=Path(tempfile.mkdtemp(prefix="astra_p2_series_",dir="/private/tmp"))
url=f"ipc://{scratch}/flops.sock"
env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",FLOPSCOPE_SERVER_URL=url)
started=time.time();samples=[];client=None
with log.open("w") as stream:
    server=subprocess.Popen([str(HERE/"openblas_runtime_env_v1/bin/python"),"-m","flopscope_server","--url",url,"--timeout","600"],env=env,stdout=stream,stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            assert server.poll() is None,"Diagnostic server exited"
            if (scratch/"flops.sock").exists():break
            time.sleep(.05)
        client=subprocess.Popen([str(GATE/".client-venv/bin/python"),"-u",str(HERE/"check_remote_series_v1.py"),"--candidate",args.candidate,"--tag",args.tag],env=env,cwd=HERE)
        print(json.dumps(dict(server_pid=server.pid,client_pid=client.pid)),flush=True)
        while client.poll() is None:
            if time.time()-started>900:
                raise TimeoutError("Diagnostic series exceeded 900 seconds")
            usage=subprocess.run(["/bin/ps","-o","rss=","-p",f"{server.pid},{client.pid}"],capture_output=True,text=True)
            if usage.returncode==0:
                rss=sum(int(value) for value in usage.stdout.split())*1024
                samples.append((time.time()-started,rss))
                if rss>10*1024**3:
                    raise MemoryError("Own diagnostic processes exceeded the 10 GiB safety stop")
            time.sleep(.5)
        code=client.returncode
    finally:
        if client is not None and client.poll() is None:
            client.terminate()
            try:client.wait(timeout=10)
            except subprocess.TimeoutExpired:client.kill();client.wait()
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait()
        record=dict(candidate=args.candidate,peak_combined_rss_bytes=max((value for _,value in samples),default=0),
                    max_single_child_rss_bytes=int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
                    samples=samples,wall_s=time.time()-started,
                    scope="Sampled combined client/server RSS on macOS OpenBLAS; not Linux cgroup memory.")
        out.write_text(json.dumps(record,indent=2)+"\n")
        print(json.dumps({key:value for key,value in record.items() if key!="samples"}),flush=True)
assert code==0,code
