from __future__ import annotations
import argparse, pickle
from pathlib import Path
import pandas as pd
MAP={"O":"openness","C":"conscientiousness","E":"extraversion","A":"agreeableness","N":"neuroticism"}

def main():
    ap=argparse.ArgumentParser(description="Convert a lawfully obtained FI annotation pickle into an ID-keyed local labels CSV.")
    ap.add_argument("--annotation",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,required=True,help="Released split manifest used to restrict/reorder the local labels.")
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    with a.annotation.open("rb") as f: data=pickle.load(f,encoding="latin1")
    missing=[v for v in MAP.values() if v not in data]
    if missing: raise KeyError(f"Annotation pickle missing expected traits: {missing}")
    manifest=pd.read_csv(a.manifest); ids=manifest["sample_id"].astype(str).tolist()
    rows=[]
    for sid in ids:
        row={"sample_id":sid}
        for short,long in MAP.items():
            if sid not in data[long]: raise KeyError(f"{sid} missing from annotation trait {long}")
            row[f"target_{short}"]=float(data[long][sid])
        rows.append(row)
    out=pd.DataFrame(rows); a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
    print(f"Wrote {len(out)} local label rows to {a.output}")
if __name__=="__main__": main()
