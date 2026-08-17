from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
TRAITS=list("OCEAN")

def main():
    ap=argparse.ArgumentParser(description="Create a public-safe prediction CSV by removing target/label columns from a local evaluation export.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a=ap.parse_args(); df=pd.read_csv(a.input)
    required=["sample_id"]+[f"pred_{t}" for t in TRAITS]
    missing=[c for c in required if c not in df.columns]
    if missing: raise KeyError(f"Missing columns: {missing}")
    out=df[required].copy(); a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
    print(f"Wrote {len(out)} rows to {a.output}; no target_* columns were retained.")
if __name__=="__main__": main()
