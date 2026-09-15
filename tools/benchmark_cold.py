"""Supplemental first-prediction versus warm latency for a cold-timer worker."""
import argparse,json,random,subprocess,statistics
p=argparse.ArgumentParser()
p.add_argument("--binary",default="build/threaded/cold-worker")
p.add_argument("--output",default="docs/private/benchmarks/threaded-cold.json")
a=p.parse_args()
from pathlib import Path
jobs=[(rows,threads,repeat) for rows in [128,512,4096] for threads in [1,8] for repeat in range(7)]
random.Random(9241).shuffle(jobs)
records=[]
for rows,threads,repeat in jobs:
    cmd=['taskset','-c','8-15',a.binary,str(rows),'64','6','float32','margin','8',str(threads),'100']
    cold,warm=map(float,subprocess.check_output(cmd,text=True).split())
    records.append(dict(rows=rows,threads=threads,repeat=repeat,cold_seconds=cold,warm_seconds=warm))
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(records,indent=2)+'\n')
for rows in [128,512,4096]:
    for threads in [1,8]:
        found=[r for r in records if r['rows']==rows and r['threads']==threads]
        print(rows,threads,*[round(statistics.median(r[key] for r in found)*1e6,1) for key in ['cold_seconds','warm_seconds']])
