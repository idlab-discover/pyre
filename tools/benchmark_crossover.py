"""Supplemental randomized warm/cold sweep (not pyperf significance data)."""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',default='build/threaded/benchmark-worker')
    p.add_argument('--output',default='docs/benchmarks/threaded-crossover.json')
    a=p.parse_args()
    jobs=[(rows,threads,repeat) for rows in [128,256,512,1024] for threads in [1,2,4,8] for repeat in range(7)]
    random.Random(9241).shuffle(jobs)
    records=[]
    for rows,threads,repeat in jobs:
        loops=100
        cmd=['taskset','-c','8-15',a.binary,str(rows),'64','6','float32','margin','8',str(threads),str(loops)]
        elapsed=float(subprocess.check_output(cmd,cwd=ROOT,text=True))
        records.append(dict(rows=rows,threads=threads,repeat=repeat,seconds=elapsed/loops))
    (ROOT/a.output).write_text(json.dumps(records,indent=2)+'\n')
    for rows in [128,256,512,1024]:
        print(rows, {threads:round(statistics.median(r['seconds'] for r in records if r['rows']==rows and r['threads']==threads)*1e6,1) for threads in [1,2,4,8]})
if __name__=='__main__':main()
