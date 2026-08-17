from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
TRAITS=list("OCEAN")

def main():
    ap=argparse.ArgumentParser(description="Apply released or locally refit DCS affine parameters to raw OCEAN predictions.")
    ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--parameters",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); df=pd.read_csv(a.predictions); pars=json.loads(a.parameters.read_text(encoding="utf-8"))["parameters"]
    out=df[["sample_id"]].copy()
    for t in TRAITS:
        x=df[f"pred_{t}"].to_numpy(float); q=pars[t]
        out[f"pred_{t}"]=np.clip(q["slope_a"]*x+q["intercept_b"],0.0,1.0)
    a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(f"Wrote {len(out)} calibrated rows to {a.output}")
if __name__=="__main__": main()
