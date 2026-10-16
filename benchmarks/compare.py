"""Compare three local kernel variants with identical input dimensions."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",default="./kernels")
    p.add_argument("--m",type=int,default=128)
    p.add_argument("--k",type=int,default=256)
    p.add_argument("--n",type=int,default=128)
    p.add_argument("--tile",type=int,default=32)
    p.add_argument("--repeats",type=int,default=7)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--output",default="results/comparison.json")
    a=p.parse_args()
    raw=subprocess.check_output([a.binary,*map(str,(a.m,a.k,a.n,a.tile,a.repeats,a.threads))],text=True)
    rows={r["method"]:r for r in csv.DictReader(raw.splitlines())}
    for r in rows.values():
        if float(r["max_abs_error"])>=0.002:raise RuntimeError("Numerical verification failed")
    baseline=float(rows["naive"]["median_ms"])
    result={"config":vars(a),"methods":rows,"speedup_vs_naive":{
        name:round(baseline/float(row["median_ms"]),3) for name,row in rows.items()
    },"note":"Local CPU timing; no fixed speedup requirement in CI"}
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
